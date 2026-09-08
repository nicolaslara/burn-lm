//! The kernel: one plane per `(lane, kv_head)`, walking that lane's block table in place.
//!
//! # Why this decomposition
//!
//! A plane owns exactly one lane, so `lengths[lane]` — and therefore the block count and every
//! per-block bound derived from it — is *plane-uniform by construction*. That single property is
//! what the whole shape buys: no loop bound and no predicate in the walk ever differs across the
//! plane, so no plane reduction ever sits under divergent control flow. Which in turn means no
//! shared memory, no `sync_cube`, no `sync_plane`, no idle-unit sentinel dance, and no dependency
//! on `Plane::NonUniformControlFlow`. The kernel needs `Plane::Ops` and nothing else.
//!
//! Within a plane, a unit owns a *cyclic* slice of `head_dim`: `d = u + i·plane_dim`. Cyclic, not
//! blocked, because at a fixed `i` the plane's units then read `plane_dim` consecutive elements —
//! one coalesced transaction per load instead of one per unit — and the output store coalesces for
//! the same reason. The unit also owns its own slice of all `n_rep` accumulators, so the epilogue
//! has no cross-unit reduction at all: each unit divides its own values and stores them.
//!
//! Grouped-query attention is structural rather than a special case: K and V are read once per key
//! position and serve all `n_rep` query heads out of registers. There is no code path where the
//! fold does not happen, so K/V traffic is `Σ_j 2·ceil(len_j/bs)·bs·num_kv_heads·head_dim·sizeof(E)`
//! — the floor, plus only the block-granularity tail.
//!
//! # Why the walk is length-gated
//!
//! `ceil(len[lane]/block_size)` blocks, not `blocks_per_lane`. At the ragged widths the scheduler
//! actually produces this is a bigger effect than removing the gather: a rectangle costs
//! `n · l_max`, a table walk costs `Σ_j len_j`. It also means the sentinel-padded tail of the block
//! table and the dead tail of a lane's last live block are *never read*, which is what retires the
//! mask instead of reimplementing it.

use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::{Shape, TensorMetadata};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeRuntime;
use cubecl::client::ComputeClient;
use cubecl::features::Plane;
use cubecl::prelude::*;

/// Load one element of a head row, with the ragged-`head_dim` tail folded to an exact zero.
///
/// `head_dim` need not be a multiple of `plane_dim` (80 and 96 are real head dims), so the last
/// unrolled step can address past the row. Both the index and the value are chosen with `select`
/// rather than a branch: the read always lands inside the row it belongs to, the out-of-range lanes
/// contribute an exact `0.0` to the dot product, and — the point — no `plane_sum` ever ends up
/// under a predicate.
#[cube]
fn read_masked<E: Float>(
    tensor: &Tensor<E>,
    base: usize,
    d: usize,
    #[comptime] head_dim: usize,
) -> f32 {
    let live = d < head_dim;
    let idx = select(live, d, 0);
    select(live, f32::cast_from(tensor[base + idx]), f32::new(0.0))
}

/// Paged attention, decode only (`seq_q == 1`).
///
/// Every buffer is contiguous by precondition (asserted on the host), so offsets are built from
/// shapes and never by dividing strides. That is deliberate: stride arithmetic under a vectorized
/// or reshaped view is the one bug class here that corrupts silently rather than panicking.
///
/// Accumulation is f32 whatever the storage dtype `E` is.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn paged_decode_kernel<E: Float, I: Int>(
    q: &Tensor<E>,           // [n, num_heads, 1, head_dim]
    k_pool: &Tensor<E>,      // [num_blocks, block_size, num_kv_heads, head_dim]
    v_pool: &Tensor<E>,      // same shape as k_pool
    block_table: &Tensor<I>, // [n * blocks_per_lane]
    lengths: &Tensor<I>,     // [n]
    out: &mut Tensor<E>,     // [n, num_heads, 1, head_dim]
    scale: f32,
    block_size: u32,
    blocks_per_lane: u32,
    num_kv_heads: u32,
    #[comptime] head_dim: usize,
    #[comptime] plane_dim: usize,
    #[comptime] dpu: usize, // ceil(head_dim / plane_dim)
    #[comptime] n_rep: usize,
    #[define(E, I)] _dtypes: [ElemType; 2],
) {
    // Who am I. `cube_count` is exactly the number of working planes, so there is no range guard
    // and no `terminate!()`: every launched unit has real work.
    let lane = CUBE_POS_Y as usize;
    let kv_head = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    let bs = block_size as usize;
    let bpl = blocks_per_lane as usize;
    let kvh = num_kv_heads as usize;
    let num_heads = kvh * n_rep;

    // `len` IS the mask. A decode lane always holds at least one position, so the first key is
    // always real: the sentinel below never survives into an `exp` of a live term, and `l` always
    // ends at or above 1.0.
    let len = u32::cast_from(lengths[lane]) as usize;

    // Per-unit register state. Zeroed explicitly — `Array::new` does allocate with a zero
    // attribute, but cubek's own kernels do not rely on it and neither does this one.
    let mut qv = Array::<f32>::new(n_rep * dpu);
    let mut acc = Array::<f32>::new(n_rep * dpu);
    let mut m = Array::<f32>::new(n_rep);
    let mut l = Array::<f32>::new(n_rep);
    let mut kv = Array::<f32>::new(dpu);
    let mut vv = Array::<f32>::new(dpu);
    let mut s = Array::<f32>::new(n_rep);

    #[unroll]
    for g in 0..n_rep {
        let head = kv_head * n_rep + g;
        let qbase = (lane * num_heads + head) * head_dim;
        #[unroll]
        for i in 0..dpu {
            let d = u + i * plane_dim;
            qv[g * dpu + i] = read_masked::<E>(q, qbase, d, head_dim);
            acc[g * dpu + i] = f32::new(0.0);
        }
        // Finite sentinel, not -inf: `exp(sentinel - s)` is exactly 0, while `-inf` minus `-inf`
        // would be NaN and WGSL rejects an infinity literal.
        m[g] = f32::new(-3.402_823_5e38);
        l[g] = f32::new(0.0);
    }

    // Walk ONLY this lane's live blocks. Both bounds below are plane-uniform.
    let n_blocks = (len + bs - 1) / bs;
    for b in 0..n_blocks {
        let blk = u32::cast_from(block_table[lane * bpl + b]) as usize;
        let base = blk * bs;
        let remaining = len - b * bs;
        // At least 1, and uniform across the plane: no per-key predicate, no masked lanes.
        let live = min(bs, remaining);

        for t in 0..live {
            let row = ((base + t) * kvh + kv_head) * head_dim;

            // K read ONCE, serving all `n_rep` query heads.
            #[unroll]
            for i in 0..dpu {
                let d = u + i * plane_dim;
                kv[i] = read_masked::<E>(k_pool, row, d, head_dim);
            }
            // One plane reduction per query head per key. `s[g]` comes back plane-uniform.
            #[unroll]
            for g in 0..n_rep {
                let mut part = f32::new(0.0);
                #[unroll]
                for i in 0..dpu {
                    part += qv[g * dpu + i] * kv[i];
                }
                s[g] = plane_sum(part) * scale;
            }

            // V read ONCE.
            #[unroll]
            for i in 0..dpu {
                let d = u + i * plane_dim;
                vv[i] = read_masked::<E>(v_pool, row, d, head_dim);
            }

            // Online softmax. `l` and `acc` are rescaled by the same alpha and take the same p, so
            // they cannot drift apart. On the first key `m` is the sentinel, so alpha is exactly 0
            // and whatever `acc` held is erased rather than merged.
            #[unroll]
            for g in 0..n_rep {
                let m_new = max(m[g], s[g]);
                let alpha = (m[g] - m_new).exp();
                let p = (s[g] - m_new).exp();
                l[g] = l[g] * alpha + p;
                #[unroll]
                for i in 0..dpu {
                    acc[g * dpu + i] = acc[g * dpu + i] * alpha + p * vv[i];
                }
                m[g] = m_new;
            }
        }
    }

    // Epilogue: no cross-unit reduction at all, and the store coalesces for the same reason the
    // loads do.
    #[unroll]
    for g in 0..n_rep {
        let head = kv_head * n_rep + g;
        let obase = (lane * num_heads + head) * head_dim;
        let inv = f32::new(1.0) / max(l[g], f32::new(1.0e-30));
        #[unroll]
        for i in 0..dpu {
            let d = u + i * plane_dim;
            if d < head_dim {
                out[obase + d] = E::cast_from(acc[g * dpu + i] * inv);
            }
        }
    }
}

/// Whether this device can run the kernel at this problem size.
///
/// Called twice on purpose, from two places that must not be allowed to disagree: the host gate
/// asks it so it can answer `None` and fall back, and [`launch`] asserts it, because reaching the
/// launch with an unsupported configuration means the gate is wrong — a bug to fail loudly on, not
/// a runtime condition to absorb.
pub(crate) fn supported<R: CubeRuntime>(
    client: &ComputeClient<R>,
    n: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> bool {
    let props = client.properties();
    let hw = &props.hardware;

    // Plane reductions are the one cooperation primitive the kernel uses.
    if !props.features.plane.contains(Plane::Ops) {
        return false;
    }
    // The kernel's whole geometry is "one plane per cube, `plane_dim` units wide", so it has to
    // know the plane size. On wgpu-over-AMD the value is a range, and on Intel it depends on the
    // kernel's own register use; neither is queryable here. Those devices take the reference path
    // and stay correct.
    if hw.plane_size_min != hw.plane_size_max {
        return false;
    }
    if hw.plane_size_max == 0 || hw.plane_size_max > hw.max_units_per_cube {
        return false;
    }
    if hw.plane_size_max > hw.max_cube_dim.0 {
        return false;
    }
    if num_kv_heads as u32 > hw.max_cube_count.0 || n as u32 > hw.max_cube_count.1 {
        return false;
    }
    let _ = head_dim;
    true
}

/// Launch the kernel, or return `None` if this device or these pools cannot take it.
///
/// The two `None` cases are different in kind and deliberately treated the same: an unsupported
/// device is permanent, a non-contiguous pool is a shape the caller could in principle fix. Both
/// mean "use the reference".
pub(crate) fn launch<R: CubeRuntime>(
    q: CubeTensor<R>,
    k_pool: CubeTensor<R>,
    v_pool: CubeTensor<R>,
    block_table: CubeTensor<R>,
    lengths: CubeTensor<R>,
    n_rep: usize,
    scale: f32,
) -> Option<CubeTensor<R>> {
    let q_shape = q.shape();
    let [n, num_heads, seq_q, head_dim] = q_shape.dims::<4>();
    let k_shape = k_pool.shape();
    let [_num_blocks, block_size, num_kv_heads, k_head_dim] = k_shape.dims::<4>();
    assert_eq!(seq_q, 1, "paged_decode is decode-only");
    assert_eq!(head_dim, k_head_dim, "q and the pool disagree on head_dim");
    assert_eq!(num_heads, num_kv_heads * n_rep, "head counts disagree");
    assert_eq!(
        v_pool.shape().dims::<4>(),
        [_num_blocks, block_size, num_kv_heads, k_head_dim],
        "the V pool must have the K pool's shape"
    );
    let table_len = block_table.shape().dims::<1>()[0];
    assert_eq!(
        table_len % n,
        0,
        "the block table must be n x blocks_per_lane"
    );
    assert_eq!(lengths.shape().dims::<1>()[0], n, "one length per lane");

    let client = q.client.clone();
    if !supported::<R>(&client, n, num_kv_heads, head_dim) {
        return None;
    }

    // The pools are read in place, addressed by shape. A strided pool view would need
    // `into_contiguous`, and copying a multi-gigabyte pool would cost more than everything this
    // kernel saves — so refuse instead. Today's pools are contiguous (`Tensor::empty` +
    // `slice_assign` + in-place `scatter_nd`), but nothing in the type system says so, and a
    // silent permanent disable here looks exactly like "the kernel didn't help".
    if !k_pool.is_contiguous() || !v_pool.is_contiguous() {
        log::warn!(
            "burn-lm paged decode: the KV pools are not contiguous, falling back to the reference \
             implementation for the rest of this process (copying the pool would cost more than \
             the kernel saves)"
        );
        return None;
    }

    let plane_dim = client.properties().hardware.plane_size_max as usize;
    let dpu = head_dim.div_ceil(plane_dim);

    // `q` arrives from a `swap_dims` in the model, so make it contiguous — it is small
    // (`n · num_heads · head_dim`). The pools are never touched.
    let q = into_contiguous(q);
    let block_table = into_contiguous(block_table);
    let lengths = into_contiguous(lengths);

    let out = empty_device_dtype::<R>(
        client.clone(),
        q.device.clone(),
        Shape::new([n, num_heads, 1, head_dim]),
        q.dtype,
    );

    // `burn_cubecl::kernel::utils::address_type!` is `pub(crate)`, so fold by hand over the public
    // `CubeTensor::required_address_type()`.
    let address_type = [
        q.required_address_type(),
        k_pool.required_address_type(),
        v_pool.required_address_type(),
        block_table.required_address_type(),
        lengths.required_address_type(),
        out.required_address_type(),
    ]
    .into_iter()
    .max()
    .unwrap_or_default();

    let f_dtype = q.dtype;
    let i_dtype = block_table.dtype;
    let blocks_per_lane = table_len / n;

    paged_decode_kernel::launch::<R>(
        &client,
        CubeCount::Static(num_kv_heads as u32, n as u32, 1),
        CubeDim::new_1d(plane_dim as u32),
        address_type,
        q.into_tensor_arg(),
        k_pool.into_tensor_arg(),
        v_pool.into_tensor_arg(),
        block_table.into_tensor_arg(),
        lengths.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        scale,
        block_size as u32,
        blocks_per_lane as u32,
        num_kv_heads as u32,
        head_dim,
        plane_dim,
        dpu,
        n_rep,
        [
            dtype_to_storage_type(f_dtype),
            dtype_to_storage_type(i_dtype),
        ],
    );

    Some(out)
}
