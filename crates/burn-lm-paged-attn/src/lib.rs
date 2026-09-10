//! A decode-only paged-attention kernel for burn-lm's paged KV cache.
//!
//! `burn-lm-paged-kv` stores each lane's keys and values in fixed blocks and, today, reads them
//! with generic tensor ops: gather the covering blocks out of the pool, transpose them, quadruple
//! the KV heads for grouped-query attention, then call the backend's fused attention under a mask.
//! Every one of those steps moves the whole working set through memory again. This crate is the
//! kernel that replaces them for the one shape that dominates serving — a single query position per
//! lane — by walking the block table itself while it reads:
//!
//! - no gather: the pool is addressed in place, `(block, offset, kv_head, d)`;
//! - no transpose: the kernel reads the pool's own token-major layout;
//! - no head expansion: one K/V read serves the whole query group from registers;
//! - no mask tensor: the walk stops at each lane's length, so padding is never read at all.
//!
//! # What this crate exposes
//!
//! Exactly one entry point, [`paged_decode`], and it answers `None` whenever it cannot help —
//! wrong shape, wrong dtype, a backend without the kernel, a device whose plane geometry the
//! kernel does not support, or the opt-in switch left at its default. The caller's contract is to
//! fall back to the reference implementation on `None`, which is what
//! `burn_lm_paged_kv::paged_attention` does. Silence is deliberate: a fallback must never be a
//! failure. It is also dangerous, because "the kernel never ran" looks exactly like "the kernel
//! did not help" from the outside — so [`kernel_launches`] counts the calls that took the kernel
//! path and the model-level tests assert it is non-zero.
//!
//! [`decode_reference_online`] is the plain-Rust twin of the kernel's recurrence: same block walk,
//! same online-softmax update, same finite sentinel, f32 throughout. It exists so the kernel can be
//! pinned from both sides — against the tensor-op reference (which is the *less* accurate side in
//! f16, since it runs its softmax in the storage dtype) and against an oracle that is accurate to
//! within the summation order of a single dot product.
//!
//! # Scope
//!
//! Decode only (`seq_q == 1`). Prefill keeps the reference path, and so does every backend that is
//! not a cubecl GPU backend. Quantized pools are out of scope: a `QFloat` tensor falls back.

#[cfg(feature = "kernel")]
mod backend;
#[cfg(feature = "kernel")]
mod kernel;

mod oracle;
mod switch;

pub use oracle::{decode_reference_online, DecodeShape};
pub use switch::{force_mode, kernel_launches, PagedAttentionMode};

use burn::tensor::{DType, Int, Tensor};

#[cfg(feature = "kernel")]
pub use backend::PagedDecodeAttention;

/// Attend one query position per lane over the paged KV pools, in place.
///
/// Returns `None` to mean "use the reference implementation". That is the answer for every
/// condition the kernel cannot serve, and the caller must treat it as routine.
///
/// - `q`: `[n, num_heads, 1, head_dim]`, RoPE already applied. May be non-contiguous (the model
///   produces it with a `swap_dims`); it is made contiguous inside the launch, which is cheap
///   because it holds one position per lane.
/// - `k_pool`, `v_pool`: the layer's two pools, `[num_blocks, block_size, num_kv_heads, head_dim]`.
///   These must be contiguous and are never copied — an `into_contiguous` on a multi-gigabyte pool
///   would cost more than everything this kernel saves, so a non-contiguous pool falls back
///   instead.
/// - `block_table`: `[n · blocks_per_lane]` block ids, each lane's covering blocks in position
///   order, short lanes padded with anything (the padding is never read). This is
///   `LanePlan::gather_idx` unchanged.
/// - `lengths`: `[n]`, each lane's sequence length *after* this round's write. This is the mask:
///   the kernel reads key positions `0..lengths[j]` of lane `j` and nothing else. It is the only
///   thing standing between a lane and its neighbour's KV, so the caller builds it from the same
///   snapshot as the block table.
/// - `n_rep`: `num_heads / num_kv_heads`.
/// - `scale`: the softmax scale, matching `AttentionModuleOptions::default()`'s `1/sqrt(head_dim)`.
///
/// # Panics
///
/// Never for an unsupported configuration — that is the `None` path. It does panic if `lengths`
/// disagrees with the shapes it was handed, because that is a caller bug that would otherwise be
/// read as an address.
pub fn paged_decode(
    q: Tensor<4>,
    k_pool: Tensor<4>,
    v_pool: Tensor<4>,
    block_table: Tensor<1, Int>,
    lengths: Tensor<1, Int>,
    n_rep: usize,
    scale: f32,
) -> Option<Tensor<4>> {
    // `Auto` cannot be answered yet — it depends on the device, which is a `q.device()` away and
    // not worth paying for on a shape the kernel would decline anyway. It is resolved below,
    // after the cheap metadata gates.
    if switch::mode() == PagedAttentionMode::Reference {
        return None;
    }

    // Metadata only: no resolve, no host sync, no allocation on the `None` path.
    let [n, num_heads, seq_q, head_dim] = q.dims();
    let [_num_blocks, _block_size, num_kv_heads, k_head_dim] = k_pool.dims();

    if seq_q != 1 {
        // Prefill. The reference path stays correct and stays the oracle.
        return None;
    }
    if !matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        // Quantized pools are out of scope; so is anything exotic.
        return None;
    }
    if k_pool.dtype() != q.dtype() || v_pool.dtype() != q.dtype() {
        // The kernel is generic over ONE float type and reads all three buffers through it, so a
        // pool stored at a different width than the query would be reinterpreted rather than
        // converted — plausible numbers out of the wrong bytes. Nothing in burn-lm produces this
        // today (the pools and the query both take the device's float dtype), which is exactly why
        // it has to be refused here rather than assumed away.
        return None;
    }
    if block_table.dtype() != lengths.dtype() {
        // Same argument, for the integer half: one `I` covers both.
        return None;
    }
    if num_heads != num_kv_heads * n_rep || head_dim != k_head_dim || head_dim == 0 {
        return None;
    }
    if v_pool.dims() != k_pool.dims() {
        return None;
    }
    if lengths.dims()[0] != n || block_table.dims()[0] % n != 0 {
        return None;
    }

    #[cfg(feature = "kernel")]
    {
        let device = q.device();
        if switch::mode() == PagedAttentionMode::Auto
            && !backend::device_defaults_to_kernel(&device)
        {
            // This backend has not earned the default. Say nothing: an opt-out that logs a warning
            // every round is noise, and `BURN_LM_PAGED_ATTENTION=kernel` is the way in.
            return None;
        }
        if !backend::device_supports(&device, n, num_kv_heads, head_dim) {
            // Say so once. This is the gate that turns "the kernel was asked for" into "the kernel
            // never ran", and it is otherwise indistinguishable from "the kernel did not help".
            static DECLINED: std::sync::Once = std::sync::Once::new();
            DECLINED.call_once(|| {
                log::warn!(
                    "burn-lm paged decode: this device declines the kernel for n={n}, \
                     num_kv_heads={num_kv_heads}, head_dim={head_dim}; using the reference \
                     implementation"
                );
            });
            return None;
        }
        let out = <burn::backend::Dispatch as PagedDecodeAttention>::paged_decode(
            q.into_dispatch(),
            k_pool.into_dispatch(),
            v_pool.into_dispatch(),
            block_table.into_dispatch(),
            lengths.into_dispatch(),
            n_rep as u32,
            scale,
        );
        switch::count_launch();
        Some(Tensor::from_dispatch(out))
    }

    #[cfg(not(feature = "kernel"))]
    {
        let _ = (q, k_pool, v_pool, block_table, lengths, scale);
        None
    }
}
