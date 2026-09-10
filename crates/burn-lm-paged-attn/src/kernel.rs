//! The kernel: planes that walk a lane's block table in place, reading the pool a vector at a time.
//!
//! # Why this decomposition
//!
//! A plane owns exactly one `(lane, kv_head)`, so `lengths[lane]` — and therefore the block count
//! and every per-block bound derived from it — is *plane-uniform by construction*. That single
//! property is what the whole shape buys: no loop bound and no predicate in the walk ever differs
//! across the plane, so no plane reduction ever sits under divergent control flow. The kernel
//! needs `Plane::Ops` and nothing else, and inside the walk there is no barrier at all.
//!
//! Within a plane, a unit owns a *cyclic* slice of the row: slot `j = u + i·plane_dim`, where a
//! slot is `vector_width` consecutive elements. Cyclic, not blocked, because at a fixed `i` the
//! plane's units then cover `plane_dim · vector_width` consecutive elements — one coalesced
//! transaction per load — and the output store coalesces for the same reason. The unit also owns
//! its own slice of all `n_rep` accumulators, so a single-plane cube has no cross-unit reduction
//! in its epilogue at all: each unit divides its own values and stores them.
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
//!
//! # Why a cube may hold several planes
//!
//! One plane per `(lane, kv_head)` is the whole grid, and at serving widths that grid is tiny: a
//! batch of 16 over 8 KV heads is 128 planes for an entire GPU. An A10G wants thousands, and with
//! only 128 it spends the round waiting on memory latency it has nothing to hide behind. So a cube
//! may hold `planes` planes, and they split the *block walk* between them cyclically — plane `p`
//! takes blocks `p, p+planes, p+2·planes, …`. Every plane still owns whole blocks, so `live` and
//! the loop bound stay plane-uniform and no plane reduction lands under divergent control flow;
//! what differs across planes is only how many trips each makes, and nothing inside the walk is
//! cube-wide.
//!
//! Each plane finishes with its own online-softmax state over its own subset of the keys, so the
//! epilogue merges them: `m` is the max over planes, and each plane's `l` and `acc` are rescaled by
//! `exp(m_p − m)` before being summed. That is one `sync_cube` for the whole kernel, at the very
//! end, and it is why the split is *within* a cube — merging partial states across cubes would
//! need a second pass over device memory, and this needs none.
//!
//! # Why there is no shared-memory K/V tile
//!
//! Because there is nothing to reuse. Staging a block's K and V into `Shared` pays off when many
//! units read the same element, and here no element is read twice by anybody: within a plane, key
//! position `t`'s slot `j` is read by exactly one unit, and the planes of a cube walk *disjoint*
//! blocks, so they do not share a key either. A tile would add a shared write, a barrier and a
//! shared read per element and remove no global traffic at all. The access it would supposedly
//! fix is already ideal — at a fixed key the plane's units cover `plane_dim · vector_width`
//! consecutive elements, which is one or two full transactions and nothing wasted.
//!
//! # Why the loads are vectorized
//!
//! Every buffer is read along `head_dim`, which is contiguous, so a unit can move
//! `vector_width` elements per instruction instead of one. The width is chosen on the host from
//! the device's own optimal load width, capped so that a row still has at least `plane_dim` slots:
//! a wider vector that left half the plane with nothing to load would trade instruction count for
//! idle units and a plane reduction over dead lanes. See [`vector_width`].

use burn::backend::cubecl::dtype_to_storage_type;
use burn::backend::{Shape, TensorMetadata};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeRuntime;
use cubecl::client::ComputeClient;
use cubecl::features::Plane;
use cubecl::prelude::*;

/// Load one slot of a head row, with the ragged tail folded to an exact zero.
///
/// A row is `slots` slots long and `slots` need not be a multiple of `plane_dim` (head_dim 80 is a
/// real head dim), so the last unrolled step can address past the row. Both the index and the
/// value are chosen without a branch: the read always lands inside the row it belongs to — slot 0,
/// which every live row has and every live row wrote — and the out-of-range lanes contribute an
/// exact `0.0` to the dot product. Which is the point: no `plane_sum` ever ends up under a
/// predicate, and no poisoned or uninitialized element is ever the thing being zeroed.
#[cube]
fn read_masked<E: Float, N: Size>(
    tensor: &Tensor<Vector<E, N>>,
    base: usize,
    slot: usize,
    #[comptime] slots: usize,
) -> Vector<f32, N> {
    let live = slot < slots;
    let idx = select(live, slot, 0);
    let value = Vector::<f32, N>::cast_from(tensor[base + idx]);
    value * Vector::new(select(live, f32::new(1.0), f32::new(0.0)))
}

/// Sum a vector's own elements. One per query head per key, on a register.
#[cube]
fn horizontal_sum<N: Size>(v: Vector<f32, N>) -> f32 {
    let mut total = f32::new(0.0);
    #[unroll]
    for k in 0..N::value() {
        total += v.extract(k);
    }
    total
}

/// Paged attention, decode only (`seq_q == 1`).
///
/// Every buffer is contiguous by precondition (asserted on the host), so offsets are built from
/// shapes and never by dividing strides. That is deliberate: stride arithmetic under a vectorized
/// or reshaped view is the one bug class here that corrupts silently rather than panicking. The
/// offsets below are in *slots*, and every row base is a multiple of `head_dim`, which the host
/// guarantees is a multiple of the vector width — so no row ever starts mid-vector.
///
/// Accumulation is f32 whatever the storage dtype `E` is.
#[cube(launch, address_type = "dynamic")]
#[allow(clippy::too_many_arguments)]
fn paged_decode_kernel<E: Float, I: Int, N: Size>(
    q: &Tensor<Vector<E, N>>,       // [n, num_heads, 1, head_dim]
    k_pool: &Tensor<Vector<E, N>>,  // [num_blocks, block_size, num_kv_heads, head_dim]
    v_pool: &Tensor<Vector<E, N>>,  // same shape as k_pool
    block_table: &Tensor<I>,        // [n * blocks_per_lane]
    lengths: &Tensor<I>,            // [n]
    out: &mut Tensor<Vector<E, N>>, // [n, num_heads, 1, head_dim]
    scale: f32,
    block_size: u32,
    blocks_per_lane: u32,
    num_kv_heads: u32,
    #[comptime] slots: usize, // head_dim / vector_width
    #[comptime] plane_dim: usize,
    #[comptime] dpu: usize,    // ceil(slots / plane_dim)
    #[comptime] planes: usize, // planes per cube, each taking every `planes`-th block
    #[comptime] n_rep: usize,
    #[define(E, I)] _dtypes: [ElemType; 2],
) {
    // Who am I. `cube_count` is exactly the number of working cubes, so there is no range guard
    // and no `terminate!()`: every launched unit has real work (or, for a plane that drew no
    // blocks, a merge contribution the epilogue is built to absorb).
    let lane = CUBE_POS_Y as usize;
    let kv_head = CUBE_POS_X as usize;
    let u = UNIT_POS_X as usize;
    // `CubeDim.x` is exactly `plane_dim`, so a unit's y index *is* its plane index. The launch
    // builds the cube dim that way and the block split below depends on it.
    let p = UNIT_POS_Y as usize;
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
    let mut qv = Array::<Vector<f32, N>>::new(n_rep * dpu);
    let mut acc = Array::<Vector<f32, N>>::new(n_rep * dpu);
    let mut m = Array::<f32>::new(n_rep);
    let mut l = Array::<f32>::new(n_rep);
    let mut kv = Array::<Vector<f32, N>>::new(dpu);
    let mut vv = Array::<Vector<f32, N>>::new(dpu);
    let mut s = Array::<f32>::new(n_rep);

    #[unroll]
    for g in 0..n_rep {
        let head = kv_head * n_rep + g;
        let qbase = (lane * num_heads + head) * slots;
        #[unroll]
        for i in 0..dpu {
            let slot = u + i * plane_dim;
            qv[g * dpu + i] = read_masked::<E, N>(q, qbase, slot, slots);
            acc[g * dpu + i] = Vector::new(f32::new(0.0));
        }
        // Finite sentinel, not -inf: `exp(sentinel - s)` is exactly 0, while `-inf` minus `-inf`
        // would be NaN and WGSL rejects an infinity literal.
        m[g] = f32::new(-3.402_823_5e38);
        l[g] = f32::new(0.0);
    }

    // Walk ONLY this lane's live blocks, and only this plane's share of them. Both bounds below
    // are plane-uniform: `p` is constant within a plane, so `b` is too.
    let n_blocks = (len + bs - 1) / bs;
    let mut b = p;
    while b < n_blocks {
        let blk = u32::cast_from(block_table[lane * bpl + b]) as usize;
        let base = blk * bs;
        let remaining = len - b * bs;
        // At least 1, and uniform across the plane: no per-key predicate, no masked lanes.
        let live = min(bs, remaining);

        for t in 0..live {
            let row = ((base + t) * kvh + kv_head) * slots;

            // K read ONCE, serving all `n_rep` query heads.
            #[unroll]
            for i in 0..dpu {
                let slot = u + i * plane_dim;
                kv[i] = read_masked::<E, N>(k_pool, row, slot, slots);
            }
            // One plane reduction per query head per key. The per-unit part is accumulated as a
            // vector and flattened once, so the vector width costs no extra plane traffic.
            // `s[g]` comes back plane-uniform.
            #[unroll]
            for g in 0..n_rep {
                let mut part = Vector::<f32, N>::new(f32::new(0.0));
                #[unroll]
                for i in 0..dpu {
                    part += qv[g * dpu + i] * kv[i];
                }
                s[g] = plane_sum(horizontal_sum::<N>(part)) * scale;
            }

            // V read ONCE. Hoisting this above the plane reductions — so the two loads issue
            // back to back and the reduction chain hides both latencies — was tried and measured
            // as exactly no change on Metal: at 320 GB/s of roughly 400 the walk is bandwidth
            // bound, not latency bound, and the planes-per-cube split already gives the scheduler
            // plenty of other warps to run. Left in algorithm order.
            #[unroll]
            for i in 0..dpu {
                let slot = u + i * plane_dim;
                vv[i] = read_masked::<E, N>(v_pool, row, slot, slots);
            }

            // Online softmax. `l` and `acc` are rescaled by the same alpha and take the same p, so
            // they cannot drift apart. On the first key `m` is the sentinel, so alpha is exactly 0
            // and whatever `acc` held is erased rather than merged.
            #[unroll]
            for g in 0..n_rep {
                let m_new = max(m[g], s[g]);
                let alpha = (m[g] - m_new).exp();
                let weight = (s[g] - m_new).exp();
                l[g] = l[g] * alpha + weight;
                let alpha_v = Vector::<f32, N>::new(alpha);
                let weight_v = Vector::<f32, N>::new(weight);
                #[unroll]
                for i in 0..dpu {
                    acc[g * dpu + i] = acc[g * dpu + i] * alpha_v + weight_v * vv[i];
                }
                m[g] = m_new;
            }
        }
        b += planes;
    }

    // Epilogue. With one plane per cube there is nothing to merge and no cross-unit reduction at
    // all: each unit divides its own accumulators and stores them, and the store coalesces for the
    // same reason the loads do.
    if comptime!(planes == 1) {
        #[unroll]
        for g in 0..n_rep {
            let head = kv_head * n_rep + g;
            let obase = (lane * num_heads + head) * slots;
            let inv = Vector::<f32, N>::new(f32::new(1.0) / max(l[g], f32::new(1.0e-30)));
            #[unroll]
            for i in 0..dpu {
                let slot = u + i * plane_dim;
                if slot < slots {
                    out[obase + slot] = Vector::<E, N>::cast_from(acc[g * dpu + i] * inv);
                }
            }
        }
    } else {
        // Several planes each hold a partial online-softmax state over a disjoint subset of this
        // lane's keys. Merge them the way the walk merges one key into a running state, except
        // both sides are now partials: take the max of the maxima, rescale each partial by
        // `exp(m_p − m)`, and sum.
        //
        // A plane that drew no blocks at all — `n_blocks < planes` — still merges correctly
        // without being special-cased: its `m` is the finite sentinel, so its `exp(m_p − m)` is
        // exactly 0 and its zeroed `l` and `acc` contribute nothing. Some plane always has work,
        // because a decode lane always holds at least one position.
        let mut part_m = Shared::<[f32]>::new_slice(planes * n_rep);
        let mut part_l = Shared::<[f32]>::new_slice(planes * n_rep);
        let mut part_acc = Shared::<[Vector<f32, N>]>::new_slice(planes * n_rep * dpu * plane_dim);

        #[unroll]
        for g in 0..n_rep {
            if u == 0 {
                part_m[p * n_rep + g] = m[g];
                part_l[p * n_rep + g] = l[g];
            }
            #[unroll]
            for i in 0..dpu {
                part_acc[((p * n_rep + g) * dpu + i) * plane_dim + u] = acc[g * dpu + i];
            }
        }
        // The kernel's only barrier, and the only reason a cube here is more than a plane.
        sync_cube();

        // Plane 0 finishes. The merge is `O(planes · head_dim)` against a walk of
        // `O(len · head_dim)`, so handing it to one plane rather than splitting it again costs
        // nothing measurable and keeps the indexing readable.
        if p == 0 {
            #[unroll]
            for g in 0..n_rep {
                let mut m_all = part_m[g];
                #[unroll]
                for other in 1..planes {
                    m_all = max(m_all, part_m[other * n_rep + g]);
                }
                let mut l_all = f32::new(0.0);
                let mut acc_all = Array::<Vector<f32, N>>::new(dpu);
                #[unroll]
                for i in 0..dpu {
                    acc_all[i] = Vector::new(f32::new(0.0));
                }
                #[unroll]
                for other in 0..planes {
                    let alpha = (part_m[other * n_rep + g] - m_all).exp();
                    l_all += part_l[other * n_rep + g] * alpha;
                    let alpha_v = Vector::<f32, N>::new(alpha);
                    #[unroll]
                    for i in 0..dpu {
                        acc_all[i] +=
                            part_acc[((other * n_rep + g) * dpu + i) * plane_dim + u] * alpha_v;
                    }
                }
                let head = kv_head * n_rep + g;
                let obase = (lane * num_heads + head) * slots;
                let inv = Vector::<f32, N>::new(f32::new(1.0) / max(l_all, f32::new(1.0e-30)));
                #[unroll]
                for i in 0..dpu {
                    let slot = u + i * plane_dim;
                    if slot < slots {
                        out[obase + slot] = Vector::<E, N>::cast_from(acc_all[i] * inv);
                    }
                }
            }
        }
    }
}

/// How many elements one unit moves per load.
///
/// Two constraints, and they pull against each other. The device wants its full load width used —
/// `io_optimized_vector_sizes` is cubecl's own answer for that, derived from `load_width` and the
/// element size, so 4 for f32 and 8 for f16 on CUDA. The kernel wants a row to keep at least
/// `plane_dim` slots, because slots are what the plane's units divide between them: at head_dim 64
/// on a 32-wide plane, a width of 4 would leave half the plane with nothing to load while the
/// `plane_sum` still ran over all 32 lanes. So the width is the largest optimal size that both
/// divides `head_dim` and leaves `head_dim / width >= plane_dim`.
///
/// The divisibility is not a nicety: every row base in the kernel is a multiple of `head_dim`, and
/// that is the whole reason a slot index can be a row base divided by the width without a row ever
/// starting mid-vector.
fn vector_width<R: CubeRuntime>(
    client: &ComputeClient<R>,
    elem_size: usize,
    head_dim: usize,
    plane_dim: usize,
) -> usize {
    client
        .io_optimized_vector_sizes(elem_size)
        .find(|&width| width <= head_dim / plane_dim.max(1) && head_dim.is_multiple_of(width))
        .unwrap_or(1)
}

/// How many planes one cube should hold, given how few cubes the grid has.
///
/// The grid is `num_kv_heads x n` cubes and nothing else, which at serving widths is far too small
/// to fill a GPU: 16 lanes over 8 KV heads is 128 planes. Splitting each lane's block walk across
/// several planes of one cube buys back the parallelism without a second pass over device memory,
/// and the cost is one `sync_cube` and a `planes x n_rep x head_dim` scratch in shared memory.
///
/// The number is bounded from four directions and the smallest wins:
///
/// - **work**: never more planes than there are blocks to hand out, or the extra planes only wait
///   at the barrier. `blocks_per_lane` is the block table's own width, so it is an upper bound on
///   every lane in the round;
/// - **cube geometry**: `planes x plane_dim` units must fit a cube, and `planes` must fit the y
///   extent of a cube;
/// - **shared memory**: the merge scratch has to fit;
/// - **diminishing returns**: past 32 planes a cube is a whole SM's worth of warps waiting on one
///   barrier, and the unrolled merge stops being free.
#[allow(clippy::too_many_arguments)]
fn planes_per_cube<R: CubeRuntime>(
    client: &ComputeClient<R>,
    n: usize,
    num_kv_heads: usize,
    blocks_per_lane: usize,
    n_rep: usize,
    dpu: usize,
    plane_dim: usize,
    vector_width: usize,
) -> usize {
    let hw = &client.properties().hardware;

    // What "enough parallelism" means. 16 planes per SM is a latency-hiding target, not an
    // occupancy limit — the point is to have several warps per scheduler with an outstanding load,
    // not to fill the machine. Devices that do not report an SM count (wgpu does not) get a flat
    // number in the same range as the GPUs this runs on.
    let target_planes = hw
        .num_streaming_multiprocessors
        .map(|sm| sm as usize * 16)
        .unwrap_or(512);
    let cubes = (n * num_kv_heads).max(1);
    let wanted = target_planes.div_ceil(cubes).max(1).next_power_of_two();

    let by_work = blocks_per_lane.max(1);
    let by_units = (hw.max_units_per_cube as usize / plane_dim).max(1);
    let by_cube_dim = (hw.max_cube_dim.1 as usize).max(1);
    // Two `[planes * n_rep]` f32 arrays plus one `[planes * n_rep * dpu * plane_dim]` array of
    // f32 vectors.
    let floats_per_plane = n_rep * dpu * plane_dim * vector_width + 2 * n_rep;
    let by_shared = (hw.max_shared_memory_size / (floats_per_plane * 4)).max(1);

    let cap = wanted
        .min(by_work)
        .min(by_units)
        .min(by_cube_dim)
        .min(by_shared)
        .min(32);
    // Floor to a power of two: `planes` drives an unrolled merge and a shared-memory stride, and a
    // ragged count buys nothing that the extra shapes of compiled kernel would not cost back.
    1usize << cap.max(1).ilog2()
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
    // The kernel's whole geometry is "planes of `plane_dim` units", so it has to know the plane
    // size. On wgpu-over-AMD the value is a range, and on Intel it depends on the kernel's own
    // register use; neither is queryable here. Those devices take the reference path and stay
    // correct.
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
    let width = vector_width::<R>(&client, q.dtype.size(), head_dim, plane_dim);
    let slots = head_dim / width;
    let dpu = slots.div_ceil(plane_dim);
    let blocks_per_lane = table_len / n;
    let planes = planes_per_cube::<R>(
        &client,
        n,
        num_kv_heads,
        blocks_per_lane,
        n_rep,
        dpu,
        plane_dim,
        width,
    );

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

    paged_decode_kernel::launch::<R>(
        &client,
        CubeCount::Static(num_kv_heads as u32, n as u32, 1),
        // `x` is exactly one plane wide, so a unit's `UNIT_POS_Y` *is* its plane index — which is
        // what lets the kernel give plane `p` its own share of the block walk without ever asking
        // the runtime which plane it is in.
        CubeDim::new_2d(plane_dim as u32, planes as u32),
        address_type,
        width,
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
        slots,
        plane_dim,
        dpu,
        planes,
        n_rep,
        [
            dtype_to_storage_type(f_dtype),
            dtype_to_storage_type(i_dtype),
        ],
    );

    Some(out)
}
