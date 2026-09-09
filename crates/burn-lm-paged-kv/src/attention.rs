//! The paged-attention contract: a kernel where there is one, generic tensor ops everywhere else.
//!
//! [`paged_attention`] is the contract, and its signature has not changed: queries, the two block
//! stores, and the round's plan in; attention output out. What has changed is that a single-token
//! round now offers the work to `burn-lm-paged-attn`'s decode kernel first, and only falls through
//! to [`paged_attention_reference`] when the kernel declines.
//!
//! The reference is not legacy code. It is:
//!
//! - the prefill path (`seq_q > 1`), which the kernel does not serve;
//! - the implementation for every backend and device the kernel does not cover;
//! - the differential oracle the kernel is tested against.
//!
//! So it stays, verbatim, and it stays correct. That is the permanent cost of having two paths —
//! and also the property that lets the kernel be deleted in one commit if its number disappoints.
//!
//! The two paths part company on one more thing, and it is not performance. The KV pool is
//! allocated uninitialized, so the reference's gathered scratch carries columns nobody ever wrote;
//! it neutralizes them before attending, while the kernel simply never reads them. `RaggedKv` is
//! the type that makes each path say which it is doing.
//!
//! Nothing outside this module may assume the reference's scratch (or any other intermediate)
//! exists; a round that took the kernel never materializes one.

use burn::tensor::{module::attention, ops::AttentionModuleOptions, Bool, Tensor};

use crate::cache::LanePlan;
use crate::kv_cache::KeyValueCache;

/// Attend `q` over the cached keys/values of the plan's lanes.
///
/// - `q`: `[n, num_heads, seq_len, head_dim]`, RoPE already applied, one row per active lane in
///   plan order.
/// - `cache`: the layer's paged K and V stores; this round's new tokens must already be written
///   (see [`KeyValueCache::write`]).
/// - `plan`: the round's addressing — prebuilt gather index, `l_max`, and the per-lane causal +
///   padding mask that hides every lane's stale tail (load-bearing: the gathered scratch contains
///   whole blocks and other lanes' ragged tails).
/// - `n_rep`: grouped-query expansion (`num_heads / num_kv_heads`); a kernel would fold this into
///   its block walk rather than materializing repeated heads.
///
/// Returns `[n, num_heads, seq_len, head_dim]`.
pub fn paged_attention(
    q: Tensor<4>,
    cache: &KeyValueCache,
    plan: &LanePlan,
    n_rep: usize,
) -> Tensor<4> {
    let [_n, _num_heads, seq_len, head_dim] = q.dims();

    if seq_len == 1 {
        // The decode kernel, if this build and this device have one. `paged_decode` answers `None`
        // for every configuration it cannot serve — that is routine, not an error, and the
        // fall-through below is the answer.
        //
        // The pool handles live and die inside this block, on purpose. `cache.write` has already
        // run for this round (its caller does that before calling here), and `write` only skips a
        // whole-pool copy while the pool handle is uniquely owned — so a handle held past this
        // point would turn the *next* round's KV write into a copy-on-write of the entire pool.
        // See `BlockStore::pool`.
        let kernel_out = {
            let (k_pool, v_pool) = cache.pools();
            // Matches `AttentionModuleOptions::default()`: scale `1/sqrt(head_dim)`, no softcap,
            // not `is_causal` — the length walk is the mask, and a single query position at the
            // end of its own history has nothing causal left to hide.
            let scale = (1.0 / (head_dim as f64).sqrt()) as f32;
            //
            // `assume_length_gated` is the kernel's assertion about the pool's uninitialized
            // memory, and it is the one consumer entitled to make it: its block walk's inner
            // bound IS the length. For block `b` of a lane it processes exactly
            // `min(block_size, len - b·block_size)` positions and stops, and it visits only the
            // blocks in that lane's own table, so it never loads an element the round did not
            // write. Nothing is being claimed about the *contents* of the dead tails — only that
            // they are not read. See `RaggedKv`.
            burn_lm_paged_attn::paged_decode(
                q.clone(),
                k_pool.assume_length_gated(),
                v_pool.assume_length_gated(),
                plan.gather_idx.clone(),
                plan.lengths.clone(),
                n_rep,
                scale,
            )
        };
        if let Some(out) = kernel_out {
            return out;
        }
    }

    paged_attention_reference(q, cache, plan, n_rep)
}

/// The reference implementation, in generic tensor ops: gather each lane's blocks into a
/// contiguous scratch, expand grouped-query heads, and run the backend's fused attention under the
/// plan's mask.
///
/// Every copy a kernel exists to remove is visible here — the block gather, the token-major to
/// head-major transpose, and the `n_rep`-fold head expansion — which is what makes this both the
/// thing to beat and the thing to be checked against.
pub fn paged_attention_reference(
    q: Tensor<4>,
    cache: &KeyValueCache,
    plan: &LanePlan,
    n_rep: usize,
) -> Tensor<4> {
    let [n, num_heads, seq_len, _] = q.dims();
    let (k, v) = cache.gather(plan);
    // The gather returns WHOLE blocks, so every column past a lane's own length is memory nobody
    // ever wrote — and the mask below cannot save this path from it. A masked column gets zero
    // attention weight, and the aggregation then computes `0 · V_dead`, which is NaN whenever the
    // uninitialized bytes happened to spell one. Zeroing the dead columns first is what makes the
    // mask sufficient again. It happens before `repeat_kv` deliberately: `n_rep` copies of a
    // neutralized head cost the same as one, and neutralizing after the expansion would cost
    // `n_rep` times as much. See `RaggedKv`.
    let k = repeat_kv(k.neutralized(plan), n_rep);
    let v = repeat_kv(v.neutralized(plan), n_rep);
    let mask = mask_over_heads(plan.mask.clone(), num_heads);
    // The mask is the one operand the attention op reads by shape rather than by content, so a
    // disagreement with q/k here is silent corruption rather than a panic: the fused kernel takes
    // the mask's row and column extents at face value and indexes them against the query's seq_q
    // and the key's seq_kv. The plan builds all three from the same round, so this only ever fires
    // for a caller that mixed a stale plan into a forward.
    debug_assert_eq!(
        mask.dims(),
        [n, num_heads, seq_len, k.dims()[2]],
        "paged_attention got a mask that does not cover q x cached-kv"
    );
    attention(q, k, v, Some(mask), None, AttentionModuleOptions::default())
}

/// Give the plan's mask the query's head count, so both implementations behind burn's attention op
/// read it the same way.
///
/// The plan stores one mask row per lane — `[n, 1, seq_len, l_max]` — because the mask has nothing
/// to say about heads: every head of a lane sees the same live columns, so keeping `num_heads`
/// identical copies would multiply the host build and the per-round upload by 32 on the reference
/// model for no information. What it cannot do is leave that size-1 axis in place at the call.
/// burn documents the mask as `[batch, num_heads, seq_q, seq_kv]`, and the two implementations
/// behind the op reach the head axis differently: the multi-kernel fallback broadcasts a size-1
/// dimension, while cubek's fused flash kernel indexes the mask through its real strides and takes
/// the head count from the *query*. The pinned cubek folds a size-1 head extent to a zero offset,
/// so today the two agree — but one release earlier it did not, and head `h` would have read
/// `h · seq_q · seq_kv` elements past a buffer holding a single head. Nothing in the contract makes
/// that agreement permanent, and while it lapses, correctness rides on which implementation
/// autotune happens to pick for our shapes. That is not a property worth owning.
///
/// Expanding costs nothing where it matters: on the cubecl backends `expand` is metadata only — the
/// same buffer handle with a zero stride on the head axis — so the `[32, 32, 1, 4096]` mask this
/// nominally asks for is never materialized, and every head still resolves to the same address
/// whether or not the consumer special-cases broadcast dimensions. That makes the zero stride
/// strictly stronger than the size-1 shape it replaces. The ndarray backend does copy on expand, so
/// the reference path pays one mask-sized allocation per layer; on the backend that exists to be
/// obviously correct rather than fast, that is the right side of the trade.
fn mask_over_heads(mask: Tensor<4, Bool>, num_heads: usize) -> Tensor<4, Bool> {
    let [n, mask_heads, seq_len, l_max] = mask.dims();
    if mask_heads == num_heads {
        return mask;
    }
    mask.expand([n, num_heads, seq_len, l_max])
}

/// Repeat each KV head `n_rep` times for grouped-query attention. Part of the reference
/// implementation only: a paged kernel reads each KV head once and serves its query group
/// in-kernel, so this materialization disappears with it.
fn repeat_kv(x: Tensor<4>, n_rep: usize) -> Tensor<4> {
    if n_rep == 1 {
        return x;
    }
    let [n, kv_heads, seq_len, head_dim] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .expand([n, kv_heads, n_rep, seq_len, head_dim])
        .reshape([n, kv_heads * n_rep, seq_len, head_dim])
}
