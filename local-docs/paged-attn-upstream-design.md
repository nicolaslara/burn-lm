# Generalising the paged decode kernel into a burn module op

Status: **design proposal, for discussion.** Nothing here has been implemented, and the
recommendation at the end is that we should not start implementing it yet.

Audience: someone who knows burn — `ModuleOps`, `burn-cubecl`, backend extensions — but has
never read `burn-lm-paged-attn`.

---

## 1. What we have today

### 1.1 The shape of the problem

`burn-lm-paged-kv` stores each sequence's K and V in fixed-size blocks drawn from one shared pool
per layer:

```
k_pool, v_pool : [num_blocks, block_size, num_kv_heads, head_dim]   (token-major)
```

A "lane" is one active sequence. Each lane owns a list of block ids covering its positions; entry
`i` of the list covers positions `[i·block_size, (i+1)·block_size)`. Lanes are ragged: in one
decode round lane 0 may hold 3 tokens and lane 7 may hold 4000.

One round's addressing is a `LanePlan` (`crates/burn-lm-paged-kv/src/cache.rs`). It carries nine
fields; the kernel uses exactly two of them.

### 1.2 The reference path

`paged_attention_reference` (`crates/burn-lm-paged-kv/src/attention.rs`) does what a framework
without a paged kernel must do:

1. **gather** each lane's covering blocks out of the pool into a dense
   `[n, kv_heads, l_max, head_dim]` scratch, where `l_max` is the longest lane in the round;
2. **neutralise** the dead columns — the pool is allocated with `Tensor::empty`, so every position
   past a lane's own length is memory nobody wrote, and a masked column still reaches the
   aggregation as `0 · V_dead`, which is NaN whenever those bytes spell one;
3. **expand** KV heads `n_rep`-fold for grouped-query attention;
4. call burn's fused `attention` module op under a `[n, num_heads, 1, l_max]` boolean mask.

Steps 1–3 move the entire working set through memory again, and step 4 pays `n · l_max` where the
real work is `Σ_j len_j`.

### 1.3 The kernel path

`burn_lm_paged_attn::paged_decode` replaces all four steps for the one shape that dominates
serving — one query position per lane. Its entire signature:

```rust
pub fn paged_decode(
    q: Tensor<4>,                  // [n, num_heads, 1, head_dim]
    k_pool: Tensor<4>,             // [num_blocks, block_size, num_kv_heads, head_dim]
    v_pool: Tensor<4>,             // same shape
    block_table: Tensor<1, Int>,   // [n * blocks_per_lane], row-major per lane
    lengths: Tensor<1, Int>,       // [n], each lane's length AFTER this round's write
    n_rep: usize,                  // num_heads / num_kv_heads
    scale: f32,
) -> Option<Tensor<4>>             // [n, num_heads, 1, head_dim], or None = "use the reference"
```

What it removes, mechanically:

- **no gather** — the pool is addressed in place as `(block, offset, kv_head, d)`;
- **no transpose** — it reads the pool's own token-major layout;
- **no head expansion** — one K/V read serves all `n_rep` query heads out of registers, so K/V
  traffic is the floor `Σ_j 2·ceil(len_j/bs)·bs·num_kv_heads·head_dim·sizeof(E)`;
- **no mask tensor** — the block walk's bound *is* the length, so padding is never read at all.
  This is also why the kernel is immune to the uninitialised-memory problem that forces step 2 on
  the reference.

Decomposition: one plane per `(lane, kv_head)`, so `lengths[lane]` and every bound derived from it
are plane-uniform by construction and no plane reduction ever sits under divergent control flow.
At serving widths that grid is tiny (16 lanes × 8 KV heads = 128 planes), so a cube may hold
several planes which split the lane's block walk cyclically and merge their partial online-softmax
states through one `sync_cube` in the epilogue. Accumulation is f32 whatever the storage dtype is.

### 1.4 Measured behaviour

Reported from the roofline bench (`bench_paged_decode_against_the_reference`, an `#[ignore]`d test
in `burn-lm-paged-kv/src/kernel_tests.rs`), A10G, f32, 8 kv_heads × n_rep 4, head_dim 64,
block_size 128: **2.0× at 128 tokens, 6.8× at 1024, 24.3× at 4096**, reaching **356 GB/s** of the
A10G's ~600 GB/s. Round time is nearly flat in context length where the reference grows linearly.
I did not re-run this (no local CUDA, and re-running it would need a push I am not permitted to
make); the in-tree comments carry the corresponding Metal and fp16 A10G numbers.

One property is worth separating from the speed, because it is a *correctness* argument: on CUDA
the kernel is **more accurate** than the tensor-op path, because it never calls matmul and so is
never TF32-promoted.

Default-on for Metal and CUDA; opt-in everywhere else.

---

## 2. The gap in burn

burn has a fused `attention` module op (`burn_tensor::module::attention`, cubek flash under
autotune, with `attention_fallback` behind it). It has nothing paged. Every burn user who builds a
serving stack will hit exactly the wall `paged_attention_reference` describes: a gather, a
transpose, an expand and a rectangular mask per layer per token.

So the question is a real one. The rest of this document is about what it would cost to answer it.

---

## 3. What is general, and what is ours

### 3.1 The interface: `LanePlan` is not the problem it looks like

`LanePlan` is our type, but the kernel never sees it. `attention.rs` passes exactly
`plan.gather_idx` and `plan.lengths`, and everything else in the plan — `lanes`, `starts`,
`tables`, `write_idx`, `blocks_per_lane`, `l_max`, `mask` — is host bookkeeping that stays on our
side of the call. The kernel's real interface is already two int tensors.

That matters because those two tensors are, to within a reshape, what the industry takes:

| | block addressing | lengths |
|---|---|---|
| **vLLM** `paged_attention_v1/v2` | `block_tables : [num_seqs, max_num_blocks_per_seq]` | `seq_lens : [num_seqs]` |
| **FlashInfer** `BatchDecodeWithPagedKVCacheWrapper` | CSR: `paged_kv_indptr : [batch+1]`, `paged_kv_indices : [nnz]` | `paged_kv_last_page_len : [batch]` |
| **ours** | `block_table : [n · blocks_per_lane]` (flat) | `lengths : [n]` |

Ours is vLLM's, flattened. The only differences are cosmetic (rank 1 vs rank 2) or a strict
generalisation in our favour (`lengths` is a token count, which carries `last_page_len`'s
information plus the block count, so a consumer needs one tensor where FlashInfer needs two).

**Proposed general signature.** Nothing in it is burn-lm-specific:

```rust
pub fn paged_attention(
    query:       Tensor<4>,      // [n, num_heads, seq_q, head_dim]
    key_cache:   Tensor<4>,      // [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: Tensor<4>,      // [num_blocks, block_size, num_kv_heads, val_dim]
    block_table: Tensor<2, Int>, // [n, max_blocks_per_seq]
    seq_lens:    Tensor<1, Int>, // [n] — key positions 0..seq_lens[j] are live
    options: PagedAttentionOptions, // scale, softcap, sliding_window, …
) -> Tensor<4>                   // [n, num_heads, seq_q, val_dim]
```

Changes from ours, and why:

- **`block_table` becomes rank 2.** `blocks_per_lane` then lives in the shape instead of being
  recovered by dividing a length — one fewer thing a caller can get wrong, and it is what vLLM
  does. Our flat layout is the same bytes; the reshape is free.
- **`n_rep` disappears.** It is `num_heads / num_kv_heads`, derivable from the two shapes. We pass
  it only because we had it.
- **`scale` becomes an options struct**, matching `AttentionModuleOptions`, so softcap and sliding
  window have somewhere to go later without a signature break.
- **`l_max`, `mask`, `lanes`, `starts`, `write_idx` do not appear at all.** They were never in the
  kernel's interface.

The CSR (FlashInfer) variant is strictly better when block counts are very ragged, since the
rectangular table wastes `n · max_blocks` entries. It is not worth v1: the waste is int32s, not KV
bytes, and our lengths already do the job the `last_page_len` half of CSR does. Worth naming as a
possible later addition.

### 3.2 What genuinely is ours and would have to go

**`device_defaults_to_kernel`** (`backend.rs`) — a hand-maintained "Metal yes, CUDA yes, the rest
no" list, justified in a 30-line comment about which hardware we measured. This is the piece that
most obviously cannot go upstream: burn already has the right mechanism, which is autotune. burn's
own `attention` registers a tunable set with roofline bounds
(`burn-cubecl/src/kernel/attention/`), and a paged op should do the same — the paged kernel and the
generic fallback as two tunables, benchmarked per shape bucket on the device that is actually
present. That deletes our per-backend table and replaces a judgement call with a measurement.

**`BURN_LM_PAGED_ATTENTION` and `kernel_launches()`** — see §4.4.

**`set_output_scale` / the epilogue error injection** — our own paranoia about proving the GPU ran
the kernel body rather than the fallback. It is a good trick and it should stay fork-local; it is
not an API.

**`RaggedKv`** — our type that tracks "this KV still contains positions nobody wrote", with
`neutralized` (zero them) and `assume_length_gated` (assert this consumer stops at each length) as
its two exits. The *type* is ours. The *obligation* is intrinsic to any paged attention op that
has a non-paged fallback, because the fallback gathers whole blocks and will reach dead memory the
kernel never touches. Upstream this becomes a documented contract on the op — "positions at or past
`seq_lens[j]` are never semantically read; implementations that materialise them must neutralise
them first" — and one line in the fallback.

---

## 4. Open questions, with recommended answers

### 4.1 May a module op decline?

**Our shape:** `paged_decode` returns `Option`, and `paged_attention` in `attention.rs` falls
through to the reference on `None`. The caller owns the fallback.

**Answer: at the op boundary, no — but the decline is still fine, it just moves inward.** This is
already precisely how burn's own attention works. `burn_cubecl::kernel::attention::attention`
tries flash and, on `Err(AttentionSetupError::InvalidConfig(_))`, degrades to
`attention_fallback` itself, with a comment saying so. The public `ModuleOps::attention` always
serves; the *implementation* declines internally.

So the port is: keep every gate we have, and put the fallback behind the same function instead of
in front of it. The public op always serves. That is strictly better than what we do, because it
also removes the trap our `Option` creates — a caller who forgets the fallback gets a
`None`-shaped bug rather than a slow but correct answer.

The cost is §4.5: the fallback has to be written in generic ops from `block_table` + `seq_lens`,
because there is no `LanePlan` upstream to hand it a prebuilt gather index and mask.

### 4.2 Decode-only?

**Our shape:** `seq_q == 1` or fall back. Prefill uses the reference path.

**Answer: propose it decode-only, and expect to have to justify that, not to have to fix it.**
vLLM shipped a decode-only paged kernel for years and ran prefill through a separate path
(varlen flash attention), because the two have genuinely different structure: decode is
bandwidth-bound with no matmul to speak of, prefill is compute-bound and wants tensor cores. Our
kernel's central design choice — one plane per `(lane, kv_head)`, so every bound is plane-uniform —
depends on there being exactly one query row. Adding prefill is not an extension of this kernel; it
is a second kernel.

Two ways to make that acceptable at the API level:

- the op takes `seq_q` in its signature and the *fallback* serves `seq_q > 1` correctly (which is
  exactly our situation today, and is how burn's attention handles any shape flash cannot take);
- or a companion paged prefill kernel arrives later, against the same `block_table` / `seq_lens`
  interface plus a query-offsets tensor.

Either way the interface above does not change. That is the important part: **decode-only is an
implementation restriction, not an interface restriction**, and we should not let it block the
interface discussion.

### 4.3 Contiguity

**Our shape:** `kernel::launch` refuses if `!k_pool.is_contiguous() || !v_pool.is_contiguous()`,
because offsets are built from shapes rather than strides. That is deliberate — stride arithmetic
under a vectorised view is the one bug class here that corrupts silently — and copying a
multi-gigabyte pool to satisfy it would cost more than the kernel saves.

Two problems with shipping that as-is.

First, it is **reported to be violated on CUDA at non-aligned `head_dim`**, where the allocator's
padded buffer means a freshly created pool does not satisfy `is_contiguous()`. The consequence is a
silent permanent fallback on those shapes — a coverage hole, not a wrong answer — but it is a hole
we would be shipping to other people.

Second, the warning has a bug we should fix regardless of upstreaming: the message says the kernel
will fall back "for the rest of this process", but the check is per-call and the `tracing::warn!`
has no `Once` guard (unlike the one in `lib.rs`, which does). Today it will log once per layer per
decode round.

**Recommended answer for a general op:** keep the precondition, but express it as a *layout
requirement on the cache argument* rather than a runtime coin-flip. Upstream, an op whose cache
argument must be contiguous is unremarkable — many are — as long as the internal fallback covers
the case. The remaining work is to make the fallback handle the strided pool by gathering (which it
must do anyway), so the decline is never visible.

### 4.4 The switch and the counter

**Our shape:** `BURN_LM_PAGED_ATTENTION={kernel,reference,auto}` read once per process, a
thread-local `force_mode` override for tests, and `kernel_launches()` — a global counter that
exists because a silent fallback is indistinguishable from a kernel that ran and did not help.

**None of it should go upstream, and burn already has the two replacements.**

- *Choosing an implementation* → a strategy enum parameter, as
  `burn_cubecl::kernel::attention::AttentionStrategy` already does
  (`FlashBlackboxAccelerated | FlashUnit | Fallback | Autotune`). A `PagedAttentionStrategy`
  with `{Paged, Fallback, Autotune}` gives tests both paths in one binary with no environment
  variable, no `OnceLock`, and no thread-local — which is what our thread-local exists to work
  around.
- *Differential testing* → burn already exports `burn_tensor::module::attention_fallback` with the
  doc comment "Exports attention fallback to test backend's attention against". The paged op should
  export `paged_attention_fallback` the same way. That is the whole of what our differential suite
  needs, and it is a better answer than ours because the comparison does not depend on a global.

What is *lost* is the third thing our counter buys: positive evidence that the device executed our
kernel body rather than something else. An explicit strategy that errors instead of silently
degrading when the kernel is named recovers most of it. The epilogue-injection self-check stays
ours.

### 4.5 The keystone: a generic fallback

Everything above depends on one deliverable that does not exist yet: `paged_attention_fallback`,
written in generic tensor ops, taking `block_table` and `seq_lens` *as device tensors* — no
`LanePlan`, no host-built mask, no host-built gather index, no host sync.

It is writable. Sketch:

- positions `pos = arange(max_blocks·block_size)`; source row
  `block_table[:, pos / block_size] · block_size + pos % block_size`; one `select` on the pool
  reshaped to `[num_blocks·block_size, kv_heads, head_dim]`;
- mask from lengths directly: `arange(l_max).unsqueeze() >= seq_lens.unsqueeze_dim(1)`, giving
  `[n, 1, 1, l_max]`, widened to the query's head count (and note `attention.rs`'s
  `mask_over_heads` comment about why relying on the size-1 broadcast is a trap);
- zero the gathered dead columns before attending (the `RaggedKv::neutralized` obligation);
- `n_rep` expand, then burn's existing `attention`.

This is our `paged_attention_reference` with the host-prebuilt pieces rebuilt on device. Two things
fall out of it for free, and they are why it is the keystone rather than a chore:

1. **Every backend gets a paged op**, including ndarray and tch, which is what makes it a *burn*
   op rather than a cubecl kernel with a wrapper.
2. **Autodiff is free.** `burn-autodiff`'s `ModuleOps::attention` is literally
   `attention_fallback::<Self>(...)` — composed of differentiable primitives, so the gradient
   derives itself. A paged op can do exactly the same, which retires the "every ModuleOp needs a
   backward pass and this one is inference-only" objection before it is raised.

### 4.6 Fusion

Today we register the launch as `OperationIr::Custom`, which is a **fusion barrier** — we pay
whatever elementwise fusion currently wraps the attention chain. As a real module op with its own
IR node, burn-fusion could order and fuse it properly. This is a concrete benefit we cannot obtain
fork-local at any effort, and it belongs in the argument for upstreaming.

---

## 5. Gating precondition: we do not know the kernel is correct at ragged `head_dim`

**A separate investigation is checking whether the kernel produces NaN at `head_dim = 17`** — a
head dim that is neither plane-aligned nor a multiple of any vector width. Until that is settled,
what we know is:

- the in-tree suite has a case `head_dims_that_straddle_the_plane_size_agree` covering `head_dim`
  ∈ {1, 17, 33, 65}, and `poisoned_dead_columns_cannot_reach_the_reference_output` runs at
  `head_dim: 17`, `block_size: 8`. These pass **on Metal**, which is where the suite runs locally.
- the differential grid otherwise covers `head_dim` ∈ {64, 80, 128} — 80 being the only value in
  the cross product that is not a multiple of a plane size.
- the measured performance numbers are all at aligned head dims (64 in the roofline bench).

So: **we know the kernel is correct at the aligned head dims we measured, and at ragged head dims
on Metal. We do not know it is correct at ragged head dims on CUDA.** That is exactly where the
padded-allocation interaction of §4.3 lives, which makes the two open items likely to be the same
item.

**This is a hard gate.** No upstream proposal should be opened, and no crate should be published,
until it is closed — either the kernel is correct at ragged head dims on every backend we claim, or
`supported()` refuses them explicitly and the fallback covers them. Shipping a kernel that returns
NaN on someone else's head dim would be worse than not shipping at all, because our fallback is
silent by design: the failure mode is wrong numbers, not a crash.

---

## 6. Staged plan

Stages 1–3 are worth doing **whether or not we upstream**, which is the main reason to like this
ordering.

**Stage 0 — close the gate.** Settle the `head_dim = 17` question on CUDA. Either fix it, or make
`supported()` decline ragged head dims and document why. Extend the differential grid to run ragged
head dims on every backend we default the kernel on, not just Metal. Also fix the missing `Once` on
the contiguity warning.

**Stage 1 — narrow our own interface to the general one.** Change `paged_decode` to take
`block_table: Tensor<2, Int>` and drop `n_rep`, deriving it from shapes. Purely local, no
behaviour change, and it makes the call site say what the interface actually is. Cheap.

**Stage 2 — write `paged_attention_fallback` against device tensors.** §4.5. Keep
`paged_attention_reference` alongside it and gate them against each other, since the existing one
is the oracle the whole suite is built on. This is the largest single piece of work and the one
that carries the most value on its own: it is what lets the op serve CPU backends, and it is what
autodiff would reduce to.

**Stage 3 — invert the fallback.** Make our entry point always serve, deciding internally, with a
strategy enum in place of the environment switch. Retire `kernel_launches()` in favour of the
strategy being an explicit argument. At the end of this stage our crate has the shape a burn op
would have, while still living in our tree.

**Stage 4 — decide.** Only now is the decision informed: we will know what the fallback costs, we
will know whether prefix caching (which is next on the roadmap and will change what the cache
layout wants) breaks the interface, and the correctness gate will be closed. Options at that point:
upstream RFC, standalone published crate, or stay fork-local.

**Stage 5 (if upstreaming) — RFC first, code second.** Open an issue describing the interface and
the decode-only restriction and ask whether burn wants a serving-shaped op in `burn-tensor` at all.
Do not arrive with a patch. The answer to that question is not ours to guess, and it determines
whether stages 1–3 were a port or just a cleanup — which is fine either way, because they are a
cleanup regardless.

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Ragged `head_dim` NaN (§5) | **blocking** | Stage 0. No proposal before it closes. |
| The interface is not stable — prefix caching is next and changes what the cache wants (shared blocks, copy-on-write, a block's refcount) | **high** | Stage 4 comes *after* prefix caching has a design. Upstreaming an interface we are about to change is the worst outcome available. |
| Upstream says no, or says yes-with-prefill | medium | Stages 1–3 are valuable standalone; the RFC-first ordering in stage 5 costs a week, not a quarter. |
| Maintaining a backend extension against burn internals (`backend_extension`, `CustomOpIr`, `CubeTensor`, `required_address_type`) across burn upgrades | medium, ongoing | This is an argument *for* upstreaming: that churn becomes someone else's. Today it is ours and we pay it on every bump. |
| The generic fallback is slow enough that autotune never picks it and it rots untested | medium | It is the differential oracle; it is exercised by construction. |
| `OperationIr::Custom` fusion barrier cost at short context | low, known | Already measured; the per-device default exists because of it. Upstreaming removes it (§4.6). |
| Our performance numbers do not transfer to hardware we have never run on (ROCm, Vulkan, Intel, sm90) | medium | Autotune, not a hand-maintained device list (§3.2). |

---

## 8. Recommendation

**Do not upstream now. Do stages 0–3, then reconsider — and when we do reconsider, the most likely
right answer is a standalone published crate, not a burn PR.**

The reasoning:

- The interface is about to move. Prefix caching is the next roadmap item and it changes what a
  block *is* — shared, refcounted, copy-on-write. An upstream op is a promise; making that promise
  one release before we learn what prefix caching wants is how you end up maintaining two
  interfaces.
- The correctness gate is open. §5 is not a formality.
- A third option beats both horns. Publishing `burn-paged-attn` to crates.io gets the kernel to
  every burn user who wants it, keeps our iteration speed, and does not ask burn to adopt a
  serving-shaped abstraction it may not want in a training-first framework. The only thing it does
  not get us is fusion (§4.6) — which is real, but is one item against a list.

**The case for upstreaming anyway, stated fairly**, because it is stronger than I would have
guessed before reading burn's attention code:

- Every structural objection I expected turned out to have a precedent already in burn. "An op
  cannot decline" — burn's own attention declines internally to `attention_fallback`. "Every
  ModuleOp needs a backward" — `burn-autodiff`'s attention is *literally* the fallback. "A
  hand-maintained device list is unacceptable" — autotune with roofline bounds already exists for
  attention and would replace ours. There is no design obstacle here, only work.
- The kernel is written in cubecl, so it is backend-generic in a way no ported CUDA kernel could
  be. That is burn's distinctive value proposition and nobody else can hand it to them. If burn
  ever wants a serving story, this is the shape it has to take, and we are the ones holding it.
- The accuracy argument is not about speed and does not depend on any benchmark: on CUDA the
  kernel avoids TF32 promotion because it never calls matmul, so it is *more* accurate than the
  path it replaces. That is a rare thing to be able to say about a fast kernel and it would carry
  weight in an RFC.
- The maintenance burden runs the other way from how it first looks. We are pinned to
  `=0.22.0-pre.3` partly because a backend extension reaches into burn internals. Upstream, that
  coupling stops being our problem.

If the discussion lands on upstreaming regardless, the ordering in §6 does not change — stages 0–3
are the prerequisite either way, and stage 5's RFC-before-patch rule is the part I would hold to
most firmly.

---

## Appendix: where to look in the code

| Thing | File |
|---|---|
| Host gate, the `Option` contract, dtype/shape gates | `crates/burn-lm-paged-attn/src/lib.rs` |
| The cubecl kernel, `supported()`, `vector_width`, `planes_per_cube`, contiguity refusal | `crates/burn-lm-paged-attn/src/kernel.rs` |
| Backend extension, fusion `CustomOpIr` registration, per-device default | `crates/burn-lm-paged-attn/src/backend.rs` |
| Env switch, thread-local override, launch counter, epilogue injection | `crates/burn-lm-paged-attn/src/switch.rs` |
| Plain-Rust twin of the recurrence (the tight oracle) | `crates/burn-lm-paged-attn/src/oracle.rs` |
| Runtime proof that the GPU ran the kernel body | `crates/burn-lm-paged-attn/src/selfcheck.rs` |
| `LanePlan`, `prepare_lanes` | `crates/burn-lm-paged-kv/src/cache.rs` |
| `RaggedKv`, the pool, gather | `crates/burn-lm-paged-kv/src/block_store.rs` |
| The call site: kernel-first, reference-behind | `crates/burn-lm-paged-kv/src/attention.rs` |
| Differential grid, poison tests, roofline bench | `crates/burn-lm-paged-kv/src/kernel_tests.rs` |
| burn's precedent for an op that declines internally | `burn-cubecl-0.22.0-pre.3/src/kernel/attention/base.rs` |
| burn's precedent for autodiff-via-fallback | `burn-autodiff-0.22.0-pre.3/src/ops/module.rs` (`fn attention`) |
