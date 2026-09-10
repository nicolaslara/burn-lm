//! Differential tests for the paged decode kernel.
//!
//! Compiled only when a backend feature turned the kernel on (`--features metal`, `--features
//! cuda`, …), because there is nothing to differ from otherwise. Every test here asserts
//! [`kernel_launches`](burn_lm_paged_attn::kernel_launches) actually moved: the fallback is silent
//! by design, and a test that only checks equivalence passes just as happily when the kernel never
//! ran.
//!
//! Three things are compared, and they are not redundant:
//!
//! - **The tensor-op reference** (`paged_attention_reference`) — the implementation in production
//!   today, and the one that must stay swappable. It is the *loose* side of the comparison: it
//!   materializes a gathered scratch, expands heads, and (on f16) runs its softmax in the storage
//!   dtype, so in f16 it is the less accurate of the two.
//! - **The host twin** (`decode_reference_online`) — the same recurrence in plain f32 Rust. This is
//!   the tight side: it differs from the kernel only in the summation order inside one dot product,
//!   so a wrong rescale (O(1) wrong) cannot hide behind its tolerance.
//! - **The kernel against itself** with poison written into memory it must never read. The
//!   tensor-op reference cannot pass that one — it gathers whole blocks — which is exactly why it
//!   is the proof that the length walk replaced the mask rather than reimplementing it.

use burn::tensor::{Device, Int, Tensor, TensorData};

use burn_lm_paged_attn::{
    decode_reference_online, force_mode, kernel_launches, DecodeShape, PagedAttentionMode,
};

use crate::attention::{paged_attention, paged_attention_reference};
// The device pair comes from one place for the whole crate: bringing the f16 device up while
// other threads already have work on the default one spins inside cubecl's channel init.
use crate::cache::{KvLayout, LanePlan, PagedKvCache};
use crate::kv_cache::KeyValueCache;
use crate::test_device::{f16_device, test_device as f32_device};

/// One case of the differential grid.
#[derive(Debug, Clone)]
struct Case {
    /// Tokens per KV block. Non-powers-of-two and `1` are in the grid on purpose: the kernel's
    /// per-block `live` count is `min(block_size, len - b·block_size)`, and a block size that
    /// never divides a length is what exercises it.
    block_size: usize,
    /// Active lanes this round.
    lengths: Vec<usize>,
    /// Query heads per KV head.
    n_rep: usize,
    /// Elements per head. 80 is in the grid because it is not a multiple of any plane size, so it
    /// is the only thing that exercises the kernel's ragged-`head_dim` guard.
    head_dim: usize,
    num_kv_heads: usize,
}

impl Case {
    /// The KV window the cache is built with. At least `block_size`, because a pool cut into
    /// blocks bigger than its own window is refused at construction — and a case whose longest
    /// lane is shorter than one block is a case worth having (it is the whole-history-in-one-block
    /// shape).
    fn max_seq_len(&self) -> usize {
        self.lengths
            .iter()
            .copied()
            .max()
            .unwrap()
            .max(2)
            .max(self.block_size)
    }
    fn n(&self) -> usize {
        self.lengths.len()
    }
    fn num_heads(&self) -> usize {
        self.num_kv_heads * self.n_rep
    }
}

/// A deterministic value stream. Not `rand`: this crate has no such dependency, and a fixed
/// sequence means a failure is reproducible from the test name alone.
struct Noise(u64);

impl Noise {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    /// A value in roughly `[-2, 2)`. Wide enough that the softmax has a real maximum to track (a
    /// flat score row would let a broken rescale pass) and narrow enough not to saturate f16.
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32;
        unit * 4.0 - 2.0
    }
    fn vec(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.next()).collect()
    }
}

/// One decode round, set up exactly as the model sets it up.
struct Round {
    cache: PagedKvCache,
    plan: LanePlan,
    /// `[n, num_heads, 1, head_dim]`, deliberately non-contiguous.
    q: Tensor<4>,
}

fn build_round(case: &Case, device: &Device, seed: u64) -> Round {
    let device = device.clone();
    let layout = KvLayout {
        n_layers: 1,
        n_kv_heads: case.num_kv_heads,
        head_dim: case.head_dim,
        max_seq_len: case.max_seq_len(),
    };
    let mut cache = PagedKvCache::with_window_per_lane(layout, case.n(), case.block_size, &device);
    let mut noise = Noise::new(seed);

    // Prefill each lane to its own length minus the one token this round writes. Lane by lane, so
    // the lanes end up at genuinely divergent positions rather than a shared prefix — which is the
    // whole point: a kernel that ignored `lengths` would still pass a test where every lane is the
    // same length.
    for (lane, &len) in case.lengths.iter().enumerate() {
        let prefill = len - 1;
        if prefill == 0 {
            continue;
        }
        let plan = cache.prepare_lanes(&[lane], prefill).unwrap();
        let shape = [1, case.num_kv_heads, prefill, case.head_dim];
        let k = Tensor::<4>::from_data(
            TensorData::new(noise.vec(shape.iter().product()), shape),
            &device,
        );
        let v = Tensor::<4>::from_data(
            TensorData::new(noise.vec(shape.iter().product()), shape),
            &device,
        );
        for layer in cache.layers_mut() {
            layer.write(&plan, k.clone(), v.clone());
        }
    }

    // The decode round itself: one new token per lane, all lanes at once.
    let lanes: Vec<usize> = (0..case.n()).collect();
    let plan = cache.prepare_lanes(&lanes, 1).unwrap();
    let kv_shape = [case.n(), case.num_kv_heads, 1, case.head_dim];
    let k = Tensor::<4>::from_data(
        TensorData::new(noise.vec(kv_shape.iter().product()), kv_shape),
        &device,
    );
    let v = Tensor::<4>::from_data(
        TensorData::new(noise.vec(kv_shape.iter().product()), kv_shape),
        &device,
    );
    for layer in cache.layers_mut() {
        layer.write(&plan, k.clone(), v.clone());
    }

    // `q` as the model hands it over: projected `[n, 1, heads, d]` then transposed, so the tensor
    // reaching the kernel is a strided view and the launch's `into_contiguous` is load-bearing.
    let q_shape = [case.n(), 1, case.num_heads(), case.head_dim];
    let q = Tensor::<4>::from_data(
        TensorData::new(noise.vec(q_shape.iter().product()), q_shape),
        &device,
    )
    .swap_dims(1, 2);

    Round { cache, plan, q }
}

fn layer(cache: &mut PagedKvCache) -> &mut KeyValueCache {
    cache.layers_mut().next().expect("one layer")
}

fn host(t: Tensor<4>) -> Vec<f32> {
    t.into_data().iter::<f32>().collect()
}

fn host_ints(t: Tensor<1, Int>) -> Vec<i32> {
    t.into_data().iter::<i32>().collect()
}

/// The largest difference between two runs, normalized by the largest magnitude in the reference.
///
/// Not a per-element relative difference: an attention output is a convex combination of V rows, so
/// individual components pass through zero and a per-element ratio there reports a huge error for
/// an absolutely tiny one. Normalizing by the output's own scale asks the question that matters —
/// "how big is the disagreement compared to the size of the answer".
fn max_norm_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "output lengths differ");
    let mut scale = 1.0e-6f32;
    for y in b {
        assert!(y.is_finite(), "non-finite reference output: {y}");
        scale = scale.max(y.abs());
    }
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        assert!(x.is_finite(), "non-finite kernel output: {x}");
        worst = worst.max((x - y).abs() / scale);
    }
    worst
}

/// The worst difference found in any single lane, and which lane it was.
///
/// [`max_norm_diff`] normalizes by the largest magnitude anywhere in the reference, which is the
/// right question for a round whose lanes are comparable — and the wrong one for a round holding a
/// 4096-token lane beside a 1-token lane. There the long lane sets the scale, and a short lane
/// could be answered entirely wrongly while the round-wide number stayed small. Scoring lane by
/// lane removes that hiding place, which matters precisely for the ragged shapes the length walk
/// exists to serve.
fn worst_lane_diff(a: &[f32], b: &[f32], n: usize) -> (usize, f32) {
    assert_eq!(a.len(), b.len(), "output lengths differ");
    let per = a.len() / n;
    let mut worst = (0usize, 0.0f32);
    for lane in 0..n {
        let d = max_norm_diff(
            &a[lane * per..(lane + 1) * per],
            &b[lane * per..(lane + 1) * per],
        );
        if d > worst.1 {
            worst = (lane, d);
        }
    }
    worst
}

/// The three implementations' outputs for one case, in order: kernel, tensor-op reference, host
/// twin. Every comparison in this file is a scoring of these three vectors; keeping the running of
/// them in one place is what makes it cheap to score the same round a different way.
fn run_three_ways(case: &Case, device: &Device, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    force_mode(PagedAttentionMode::Kernel);
    let launches_before = kernel_launches();

    let Round {
        mut cache, plan, q, ..
    } = build_round(case, device, seed);
    let n_rep = case.n_rep;

    // The kernel, through the production seam.
    let out_kernel = {
        let layer = cache.layers_mut().next().unwrap();
        host(paged_attention(q.clone(), layer, &plan, n_rep))
    };
    assert!(
        kernel_launches() > launches_before,
        "the kernel never ran for {case:?}: the availability gate declined, so this test would \
         otherwise be comparing the reference against itself"
    );

    // The tensor-op reference, same plan, same pools.
    let out_ref = {
        let layer = cache.layers_mut().next().unwrap();
        host(paged_attention_reference(q.clone(), layer, &plan, n_rep))
    };

    // The host twin, from the pools' actual bytes.
    let (k_pool, v_pool) = layer(&mut cache).pools();
    let (k_pool, v_pool) = (k_pool.assume_length_gated(), v_pool.assume_length_gated());
    let k_host: Vec<f32> = k_pool.into_data().iter::<f32>().collect();
    let v_host: Vec<f32> = v_pool.into_data().iter::<f32>().collect();
    let q_host: Vec<f32> = q.clone().into_data().iter::<f32>().collect();
    let table = host_ints(plan.gather_idx.clone());
    let lens = host_ints(plan.lengths.clone());
    let out_twin = decode_reference_online(
        &q_host,
        &k_host,
        &v_host,
        &table,
        &lens,
        DecodeShape {
            n: case.n(),
            num_kv_heads: case.num_kv_heads,
            n_rep,
            head_dim: case.head_dim,
            block_size: case.block_size,
            blocks_per_lane: plan.blocks_per_lane,
            scale: (1.0 / (case.head_dim as f64).sqrt()) as f32,
        },
    );

    (out_kernel, out_ref, out_twin)
}

/// Run one case both ways and against the host twin.
///
/// Returns `(kernel-vs-reference, kernel-vs-twin)` as maximum relative differences over the whole
/// round.
fn compare(case: &Case, device: &Device, seed: u64) -> (f32, f32) {
    let (out_kernel, out_ref, out_twin) = run_three_ways(case, device, seed);
    (
        max_norm_diff(&out_kernel, &out_ref),
        max_norm_diff(&out_kernel, &out_twin),
    )
}

/// As [`compare`], scored per lane: `((lane, diff) vs reference, (lane, diff) vs twin)`.
fn compare_per_lane(case: &Case, device: &Device, seed: u64) -> ((usize, f32), (usize, f32)) {
    let n = case.n();
    let (out_kernel, out_ref, out_twin) = run_three_ways(case, device, seed);
    (
        worst_lane_diff(&out_kernel, &out_ref, n),
        worst_lane_diff(&out_kernel, &out_twin, n),
    )
}

/// [`compare_per_lane`] with this file's standard f32 tolerances applied.
fn check_per_lane(case: &Case, seed: u64, label: &str) {
    let (r, t) = compare_per_lane(case, &f32_device(), seed);
    println!(
        "{label}: vs reference lane {} {:e}, vs twin lane {} {:e}",
        r.0, r.1, t.0, t.1
    );
    assert!(
        r.1 < 1.0e-5,
        "{label}: kernel vs tensor-op reference, lane {} -> {:e}",
        r.0,
        r.1
    );
    assert!(
        t.1 < 5.0e-6,
        "{label}: kernel vs host twin, lane {} -> {:e}",
        t.0,
        t.1
    );
}

/// The grid. `block_size` and the lane count are runtime scalars in the kernel, so they are free to
/// vary; `head_dim`, `n_rep` and the dtype are comptime, so each combination is a separate
/// compilation and the grid stays deliberately small on those axes. A real workload compiles one
/// variant.
fn grid() -> Vec<Case> {
    let mut cases = Vec::new();

    // The lane count is a runtime scalar (it is the cube grid's y extent), so it varies on its own
    // axis rather than inside the comptime cross product below. One lane means eight cubes on a
    // GPU with dozens of cores — the configuration the design is weakest at and the one where a
    // missing cube would be least visible — and 32 is a full serving width.
    for &n in &[1usize, 2, 3, 8, 32] {
        cases.push(Case {
            block_size: 16,
            lengths: (0..n).map(|j| 1 + j * 7 % 61).collect(),
            n_rep: 4,
            head_dim: 64,
            num_kv_heads: 8,
        });
    }

    // `n_rep` 8 (one K/V head feeding eight query heads) is the widest group any of the Llama
    // configurations use; it is here once rather than in the product because it is comptime and
    // each value is its own compilation.
    cases.push(Case {
        block_size: 32,
        lengths: vec![1, 32, 33, 103],
        n_rep: 8,
        head_dim: 64,
        num_kv_heads: 1,
    });

    for &block_size in &[1usize, 3, 16, 32, 128] {
        for &n_rep in &[1usize, 4] {
            // 64 is the reference model's; 80 is not a multiple of any plane size, so it is the
            // only thing that exercises the ragged-`head_dim` guard; 128 is four cyclic slices per
            // unit at plane size 32, the deepest unroll in the grid.
            for &head_dim in &[64usize, 80, 128] {
                for &num_kv_heads in &[1usize, 8] {
                    // Ragged on purpose, with a lane at the shortest possible decode (1) and a
                    // lane at the longest, in the same round, plus lengths that straddle every
                    // block edge.
                    cases.push(Case {
                        block_size,
                        lengths: vec![1, block_size, block_size + 1, 3 * block_size + 7, 129],
                        n_rep,
                        head_dim,
                        num_kv_heads,
                    });
                }
            }
        }
    }
    cases
}

/// f32: the kernel accumulates in f32 and so does everything it is compared against, so the only
/// error is reassociation inside a `head_dim`-term dot product and the reference's own softmax
/// rounding. Observed on an M1 Max (Metal, plane size 32) over the whole grid: 5.7e-7 against the
/// tensor-op reference, 6.7e-7 against the host twin. The tolerances below leave roughly an order
/// of magnitude — enough for a different plane width to reassociate differently, nowhere near
/// enough to hide a wrong rescale, which is O(1) wrong.
#[test]
fn kernel_matches_the_reference_and_the_host_twin_in_f32() {
    let mut worst_ref = 0.0f32;
    let mut worst_twin = 0.0f32;
    for (i, case) in grid().into_iter().enumerate() {
        let (d_ref, d_twin) = compare(&case, &f32_device(), 0xC0FFEE + i as u64);
        assert!(
            d_ref < 1.0e-5,
            "kernel vs tensor-op reference: {d_ref:e} for {case:?}"
        );
        assert!(
            d_twin < 5.0e-6,
            "kernel vs host twin: {d_twin:e} for {case:?}"
        );
        worst_ref = worst_ref.max(d_ref);
        worst_twin = worst_twin.max(d_twin);
    }
    println!("f32 worst relative difference: vs reference {worst_ref:e}, vs twin {worst_twin:e}");
}

/// f16 storage, f32 accumulation. The tolerance against the tensor-op reference is loose because
/// the *reference* is the inaccurate side here: it runs its softmax in f16. The tolerance against
/// the host twin stays much tighter, and it is the one that would catch a rescale bug — it is loose
/// only by the f16 rounding of the stored K/V the twin reads back. Observed on the same machine:
/// 1.0e-3 against the reference, 2.5e-4 against the twin.
#[cfg(all(feature = "metal", feature = "wgpu"))]
#[test]
fn kernel_matches_the_reference_and_the_host_twin_in_f16() {
    let mut worst_ref = 0.0f32;
    let mut worst_twin = 0.0f32;
    for (i, case) in grid().into_iter().enumerate() {
        let (d_ref, d_twin) = compare(&case, &f16_device(), 0xBEEF + i as u64);
        assert!(
            d_ref < 2.0e-2,
            "kernel vs tensor-op reference (f16): {d_ref:e} for {case:?}"
        );
        assert!(
            d_twin < 5.0e-3,
            "kernel vs host twin (f16): {d_twin:e} for {case:?}"
        );
        worst_ref = worst_ref.max(d_ref);
        worst_twin = worst_twin.max(d_twin);
    }
    println!("f16 worst relative difference: vs reference {worst_ref:e}, vs twin {worst_twin:e}");
}

/// The degenerate configuration: one block spans a lane's whole context, which is the old
/// per-lane slab layout. It has to keep working, because it is a configuration of the same type
/// rather than a separate code path.
#[test]
fn kernel_handles_the_unpaged_block_size() {
    let case = Case {
        block_size: 256,
        lengths: vec![1, 7, 200, 256],
        n_rep: 4,
        head_dim: 64,
        num_kv_heads: 2,
    };
    let (d_ref, d_twin) = compare(&case, &f32_device(), 7);
    assert!(d_ref < 1.0e-5, "vs reference: {d_ref:e}");
    assert!(d_twin < 5.0e-6, "vs twin: {d_twin:e}");
}

/// A shuffled, non-monotonic block table. The kernel must follow the table it is handed and
/// nothing else — a lane whose blocks were handed out in a strange order (as a recycled pool
/// hands them out) must read them in table order, not in block-id order.
#[test]
fn kernel_follows_a_non_monotonic_block_table() {
    // Free and re-allocate lanes so the pool hands out descending block ids to the long lane.
    let dev = f32_device();
    let layout = KvLayout {
        n_layers: 1,
        n_kv_heads: 2,
        head_dim: 64,
        max_seq_len: 64,
    };
    let mut cache = PagedKvCache::with_window_per_lane(layout, 4, 8, &dev);
    // Churn: fill and free so the free stack is out of order.
    for lane in 0..4 {
        cache.prepare_lanes(&[lane], 40).unwrap();
    }
    for lane in 0..4 {
        cache.reset_lane(lane);
    }

    let mut noise = Noise::new(99);
    let lengths = [37usize, 5, 24, 1];
    for (lane, &len) in lengths.iter().enumerate() {
        let prefill = len - 1;
        if prefill == 0 {
            continue;
        }
        let plan = cache.prepare_lanes(&[lane], prefill).unwrap();
        let shape = [1, 2, prefill, 64];
        let k = Tensor::<4>::from_data(
            TensorData::new(noise.vec(shape.iter().product()), shape),
            &dev,
        );
        for l in cache.layers_mut() {
            l.write(&plan, k.clone(), k.clone());
        }
    }
    let plan = cache.prepare_lanes(&[0, 1, 2, 3], 1).unwrap();
    let table = host_ints(plan.gather_idx.clone());
    assert!(
        table.windows(2).any(|w| w[1] < w[0]),
        "this test is only meaningful with a descending step somewhere in the table: {table:?}"
    );
    let kv = Tensor::<4>::from_data(TensorData::new(noise.vec(4 * 2 * 64), [4, 2, 1, 64]), &dev);
    for l in cache.layers_mut() {
        l.write(&plan, kv.clone(), kv.clone());
    }
    let q = Tensor::<4>::from_data(TensorData::new(noise.vec(4 * 8 * 64), [4, 1, 8, 64]), &dev)
        .swap_dims(1, 2);

    force_mode(PagedAttentionMode::Kernel);
    let before = kernel_launches();
    let out_kernel = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention(q.clone(), l, &plan, 4))
    };
    assert!(kernel_launches() > before, "the kernel never ran");
    let out_ref = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention_reference(q.clone(), l, &plan, 4))
    };
    let d = max_norm_diff(&out_kernel, &out_ref);
    assert!(d < 1.0e-5, "shuffled table: {d:e}");
}

/// The test the tensor-op reference cannot pass.
///
/// NaN and `1e30` are written into two places the kernel must never read: the zeroed sentinel block
/// that pads every short lane's table, and the dead tail of each lane's last live block past its
/// own length. The kernel's output must be **bit-identical** to the clean run — not close, identical
/// — because it should not have touched those bytes at all. The reference gathers whole blocks and
/// masks afterwards, so poisoning the sentinel poisons its softmax input; this is the assertion
/// that converts "the length walk replaced the mask" from a claim into a fact.
#[test]
fn poisoned_padding_cannot_reach_the_kernel_output() {
    let case = Case {
        block_size: 8,
        lengths: vec![1, 5, 8, 19],
        n_rep: 4,
        head_dim: 64,
        num_kv_heads: 2,
    };
    let dev = f32_device();

    force_mode(PagedAttentionMode::Kernel);
    let clean = {
        let Round {
            mut cache, plan, q, ..
        } = build_round(&case, &dev, 4242);
        let before = kernel_launches();
        let l = cache.layers_mut().next().unwrap();
        let out = host(paged_attention(q, l, &plan, case.n_rep));
        assert!(kernel_launches() > before, "the kernel never ran");
        out
    };

    let poisoned = {
        let Round {
            mut cache, plan, q, ..
        } = build_round(&case, &dev, 4242);
        let n_kv = case.num_kv_heads;
        let d = case.head_dim;

        // (a) the sentinel block every short lane's table is padded with.
        let sentinel_rows = Tensor::<4>::from_data(
            TensorData::new(
                (0..case.block_size * n_kv * d)
                    .map(|i| if i % 2 == 0 { f32::NAN } else { 1.0e30 })
                    .collect::<Vec<f32>>(),
                [1, n_kv, case.block_size, d],
            ),
            &dev,
        );
        for l in cache.layers_mut() {
            l.write_lanes(
                &[vec![0u32]],
                &[0],
                sentinel_rows.clone(),
                sentinel_rows.clone(),
            );
        }

        // (b) the dead tail of each lane's last live block, past that lane's own length.
        let tables = plan.tables.clone();
        for (lane, &len) in case.lengths.iter().enumerate() {
            let tail = case.block_size - (len % case.block_size);
            if tail == case.block_size {
                continue; // the lane ends exactly on a block edge; there is no tail
            }
            let rows = Tensor::<4>::from_data(
                TensorData::new(
                    (0..tail * n_kv * d)
                        .map(|i| if i % 3 == 0 { f32::NAN } else { -1.0e30 })
                        .collect::<Vec<f32>>(),
                    [1, n_kv, tail, d],
                ),
                &dev,
            );
            for l in cache.layers_mut() {
                l.write_lanes(
                    std::slice::from_ref(&tables[lane]),
                    &[len],
                    rows.clone(),
                    rows.clone(),
                );
            }
        }

        // Non-vacuity: the poison has to be in the pool for its absence from the output to mean
        // anything. A version of this test whose writes silently went nowhere would look exactly
        // as green as a kernel that genuinely never reads past a length.
        {
            let (k_pool, v_pool) = layer(&mut cache).pools();
            let (k_pool, v_pool) = (k_pool.assume_length_gated(), v_pool.assume_length_gated());
            let poisoned_cells = k_pool
                .into_data()
                .iter::<f32>()
                .chain(v_pool.into_data().iter::<f32>())
                .filter(|x: &f32| !x.is_finite())
                .count();
            assert!(
                poisoned_cells > 0,
                "the poison never reached the pools, so this test proves nothing"
            );
        }

        let l = cache.layers_mut().next().unwrap();
        host(paged_attention(q, l, &plan, case.n_rep))
    };

    assert_eq!(
        clean.len(),
        poisoned.len(),
        "the two runs must produce the same shape"
    );
    let mismatches = clean
        .iter()
        .zip(&poisoned)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        mismatches,
        0,
        "{mismatches} of {} outputs changed when padding was poisoned: the kernel read memory \
         past a lane's length",
        clean.len()
    );
}

/// The test the tensor-op reference could not pass until the pool's dead columns became a type.
///
/// Same poison, other implementation. The kernel is immune by construction — it never reads past a
/// length — but the reference gathers whole blocks, and a mask alone cannot save it: a masked
/// column gets zero attention weight, and the aggregation then computes `0 · V_dead`, which is NaN
/// whenever `V_dead` is. That is not a hypothetical about hostile input. `BlockStore::new`
/// allocates with `Tensor::empty`, so on the first use of every block the dead tail past a lane's
/// length *is* arbitrary bit patterns, and one in a few hundred of those is a NaN. Before
/// `RaggedKv::neutralized` this failed intermittently on a plain differential run, at 408
/// non-finite outputs of 544 on the run that pinned it down.
///
/// What is asserted is the two halves of the fix: the reference's output is finite, and it is
/// still the right answer — it agrees with the kernel, which read none of those bytes at all.
///
/// Swap the `neutralized` calls in `attention::paged_attention_reference` for
/// `assume_length_gated` and this fails immediately; that is the check that keeps it honest.
#[test]
fn poisoned_dead_columns_cannot_reach_the_reference_output() {
    // head_dim 17 and block_size 8 leave every lane a dead tail and none of them plane-aligned.
    let case = Case {
        block_size: 8,
        lengths: vec![1, 5, 9, 20],
        n_rep: 4,
        head_dim: 17,
        num_kv_heads: 2,
    };
    let dev = f32_device();
    let Round {
        mut cache, plan, q, ..
    } = build_round(&case, &dev, 777);
    let n_kv = case.num_kv_heads;
    let d = case.head_dim;

    // Exactly what uninitialized memory is free to contain, written where uninitialized memory
    // actually sits: the sentinel block that pads every short lane's table, and the dead tail of
    // each lane's last live block past that lane's own length.
    let sentinel_rows = Tensor::<4>::from_data(
        TensorData::new(
            vec![f32::NAN; case.block_size * n_kv * d],
            [1, n_kv, case.block_size, d],
        ),
        &dev,
    );
    for l in cache.layers_mut() {
        l.write_lanes(
            &[vec![0u32]],
            &[0],
            sentinel_rows.clone(),
            sentinel_rows.clone(),
        );
    }
    let tables = plan.tables.clone();
    for (lane, &len) in case.lengths.iter().enumerate() {
        let tail = case.block_size - (len % case.block_size);
        if tail == case.block_size {
            continue; // the lane ends on a block edge; there is no tail
        }
        let rows = Tensor::<4>::from_data(
            TensorData::new(vec![f32::NAN; tail * n_kv * d], [1, n_kv, tail, d]),
            &dev,
        );
        for l in cache.layers_mut() {
            l.write_lanes(
                std::slice::from_ref(&tables[lane]),
                &[len],
                rows.clone(),
                rows.clone(),
            );
        }
    }

    // Non-vacuity: the NaN has to be in the pool for its absence from the output to mean anything.
    {
        let (k_pool, v_pool) = layer(&mut cache).pools();
        let poisoned_cells = k_pool
            .assume_length_gated()
            .into_data()
            .iter::<f32>()
            .chain(v_pool.assume_length_gated().into_data().iter::<f32>())
            .filter(|x: &f32| !x.is_finite())
            .count();
        assert!(
            poisoned_cells > 0,
            "the NaN never reached the pools, so this test proves nothing"
        );
    }

    force_mode(PagedAttentionMode::Kernel);
    let launches_before = kernel_launches();
    let out_kernel = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention(q.clone(), l, &plan, case.n_rep))
    };
    assert!(kernel_launches() > launches_before, "the kernel never ran");

    let out_ref = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention_reference(q, l, &plan, case.n_rep))
    };
    let non_finite = out_ref.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "{non_finite} of {} reference outputs are non-finite: the gathered scratch's dead columns \
         reached the value aggregation, where `0 · NaN` is NaN however well the score was masked",
        out_ref.len()
    );

    // ...and it is still the right answer, not merely a finite one.
    let (lane, diff) = worst_lane_diff(&out_ref, &out_kernel, case.n());
    assert!(
        diff < 1.0e-5,
        "reference vs kernel under poison: lane {lane} -> {diff:e}"
    );
}

/// A lane at `len == 1`, the shortest possible decode: `alpha` is exactly 0 on the only iteration,
/// `l` ends at exactly 1.0, and the output is exactly that key's value row. Checked against the
/// stored V directly rather than against another implementation, because the answer is known.
#[test]
fn a_single_key_lane_returns_its_value_row_exactly() {
    let dev = f32_device();
    let layout = KvLayout {
        n_layers: 1,
        n_kv_heads: 2,
        head_dim: 64,
        max_seq_len: 16,
    };
    let mut cache = PagedKvCache::with_window_per_lane(layout, 1, 4, &dev);
    let plan = cache.prepare_lanes(&[0], 1).unwrap();
    let mut noise = Noise::new(1234);
    let v_vals = noise.vec(2 * 64);
    let k = Tensor::<4>::from_data(TensorData::new(noise.vec(2 * 64), [1, 2, 1, 64]), &dev);
    let v = Tensor::<4>::from_data(TensorData::new(v_vals.clone(), [1, 2, 1, 64]), &dev);
    for l in cache.layers_mut() {
        l.write(&plan, k.clone(), v.clone());
    }
    let q = Tensor::<4>::from_data(TensorData::new(noise.vec(8 * 64), [1, 1, 8, 64]), &dev)
        .swap_dims(1, 2);

    force_mode(PagedAttentionMode::Kernel);
    let before = kernel_launches();
    let l = cache.layers_mut().next().unwrap();
    let out = host(paged_attention(q, l, &plan, 4));
    assert!(kernel_launches() > before, "the kernel never ran");

    // Every one of the 8 query heads must come back as its group's V row, whatever its query was.
    for head in 0..8 {
        let kv_head = head / 4;
        for d in 0..64 {
            let got = out[head * 64 + d];
            let want = v_vals[kv_head * 64 + d];
            assert!(
                (got - want).abs() <= 1.0e-6 * want.abs().max(1.0e-3),
                "head {head} dim {d}: {got} != {want}"
            );
        }
    }
}

/// The fusion-aliasing gate.
///
/// `FusionTensor::into_ir` marks a uniquely-owned handle `ReadWrite`, which invites fusion to hand
/// a kernel that input's buffer as a writable in-place target. For a KV pool that is silent cache
/// corruption, and it would show up as a model that degrades over a conversation rather than as a
/// failure. The fusion impl clones every pool handle before `into_ir` to force `ReadOnly`; this is
/// the test that fails if those clones are ever "simplified" away.
#[test]
fn pool_bytes_are_unchanged_by_a_kernel_decode() {
    let case = Case {
        block_size: 8,
        lengths: vec![3, 8, 17],
        n_rep: 4,
        head_dim: 64,
        num_kv_heads: 2,
    };
    let Round {
        mut cache, plan, q, ..
    } = build_round(&case, &f32_device(), 555);

    let before = {
        let (k, v) = layer(&mut cache).pools();
        let (k, v) = (k.assume_length_gated(), v.assume_length_gated());
        (
            k.into_data().iter::<f32>().collect::<Vec<f32>>(),
            v.into_data().iter::<f32>().collect::<Vec<f32>>(),
        )
    };

    force_mode(PagedAttentionMode::Kernel);
    let launches = kernel_launches();
    {
        let l = cache.layers_mut().next().unwrap();
        let _ = host(paged_attention(q, l, &plan, case.n_rep));
    }
    assert!(kernel_launches() > launches, "the kernel never ran");

    let after = {
        let (k, v) = layer(&mut cache).pools();
        let (k, v) = (k.assume_length_gated(), v.assume_length_gated());
        (
            k.into_data().iter::<f32>().collect::<Vec<f32>>(),
            v.into_data().iter::<f32>().collect::<Vec<f32>>(),
        )
    };

    assert_eq!(
        before.0, after.0,
        "the K pool changed across a paged_decode: fusion was offered it as a writable in-place \
         target (see the `clone().into_ir()` calls in burn-lm-paged-attn's fusion impl)"
    );
    assert_eq!(
        before.1, after.1,
        "the V pool changed across a paged_decode"
    );
}

/// Two consecutive rounds through the production seam: write, attend with the kernel, write again,
/// attend again. This is the shape the `pools()` contract exists for — a pool handle that outlived
/// its round would turn the second round's KV write into a copy-on-write of the whole pool, and
/// the second round's *output* would then be computed against a pool that never saw the write.
/// Comparing the second round against the reference is what catches that.
#[test]
fn a_second_round_sees_the_first_rounds_writes() {
    let dev = f32_device();
    let layout = KvLayout {
        n_layers: 1,
        n_kv_heads: 2,
        head_dim: 64,
        max_seq_len: 32,
    };
    let mut cache = PagedKvCache::with_window_per_lane(layout, 2, 8, &dev);
    let mut noise = Noise::new(31337);
    force_mode(PagedAttentionMode::Kernel);

    cache.prepare_lanes(&[0], 5).unwrap();
    cache.prepare_lanes(&[1], 11).unwrap();

    for round in 0..2 {
        let plan = cache.prepare_lanes(&[0, 1], 1).unwrap();
        let kv =
            Tensor::<4>::from_data(TensorData::new(noise.vec(2 * 2 * 64), [2, 2, 1, 64]), &dev);
        for l in cache.layers_mut() {
            l.write(&plan, kv.clone(), kv.clone());
        }
        let q = Tensor::<4>::from_data(TensorData::new(noise.vec(2 * 8 * 64), [2, 1, 8, 64]), &dev)
            .swap_dims(1, 2);

        let before = kernel_launches();
        let out_kernel = {
            let l = cache.layers_mut().next().unwrap();
            host(paged_attention(q.clone(), l, &plan, 4))
        };
        assert!(
            kernel_launches() > before,
            "round {round}: the kernel never ran"
        );
        let out_ref = {
            let l = cache.layers_mut().next().unwrap();
            host(paged_attention_reference(q.clone(), l, &plan, 4))
        };
        let d = max_norm_diff(&out_kernel, &out_ref);
        assert!(d < 1.0e-5, "round {round}: {d:e}");
    }
}

/// Large scores. With `q` and `k` scaled up until the pre-softmax scores reach ~1e4, a two-pass
/// softmax that exponentiated before subtracting a maximum would overflow; the online rescale must
/// not, because it subtracts the running maximum before every `exp`. What is asserted is that the
/// output stays finite and stays close to the twin, which computes the same recurrence in f64-free
/// f32 on the host.
#[test]
fn extreme_scores_do_not_overflow_the_online_softmax() {
    let dev = f32_device();
    let layout = KvLayout {
        n_layers: 1,
        n_kv_heads: 1,
        head_dim: 64,
        max_seq_len: 16,
    };
    let mut cache = PagedKvCache::with_window_per_lane(layout, 1, 4, &dev);
    let mut noise = Noise::new(2718);

    // A prefill of 9 positions whose K rows are ~50x the usual magnitude: with scale 1/8 and
    // head_dim 64, the pre-softmax scores land around 1e4.
    let big = |noise: &mut Noise, len: usize| {
        let data: Vec<f32> = (0..len).map(|_| noise.next() * 50.0).collect();
        data
    };
    let plan = cache.prepare_lanes(&[0], 9).unwrap();
    let shape = [1usize, 1, 9, 64];
    let count = shape.iter().product::<usize>();
    let k = Tensor::<4>::from_data(TensorData::new(big(&mut noise, count), shape), &dev);
    let v = Tensor::<4>::from_data(TensorData::new(big(&mut noise, count), shape), &dev);
    for l in cache.layers_mut() {
        l.write(&plan, k.clone(), v.clone());
    }

    let plan = cache.prepare_lanes(&[0], 1).unwrap();
    let kv = Tensor::<4>::from_data(TensorData::new(big(&mut noise, 64), [1, 1, 1, 64]), &dev);
    for l in cache.layers_mut() {
        l.write(&plan, kv.clone(), kv.clone());
    }
    let q = Tensor::<4>::from_data(TensorData::new(big(&mut noise, 64), [1, 1, 1, 64]), &dev)
        .swap_dims(1, 2);

    force_mode(PagedAttentionMode::Kernel);
    let before = kernel_launches();
    let out = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention(q.clone(), l, &plan, 1))
    };
    assert!(kernel_launches() > before, "the kernel never ran");
    assert!(
        out.iter().all(|x| x.is_finite()),
        "the online softmax produced a non-finite value at large scores"
    );

    // ...and it is still the right answer, against the host twin.
    let (k_pool, v_pool) = layer(&mut cache).pools();
    let (k_pool, v_pool) = (k_pool.assume_length_gated(), v_pool.assume_length_gated());
    let out_twin = decode_reference_online(
        &q.into_data().iter::<f32>().collect::<Vec<f32>>(),
        &k_pool.into_data().iter::<f32>().collect::<Vec<f32>>(),
        &v_pool.into_data().iter::<f32>().collect::<Vec<f32>>(),
        &host_ints(plan.gather_idx.clone()),
        &host_ints(plan.lengths.clone()),
        DecodeShape {
            n: 1,
            num_kv_heads: 1,
            n_rep: 1,
            head_dim: 64,
            block_size: 4,
            blocks_per_lane: plan.blocks_per_lane,
            scale: 0.125,
        },
    );
    let d = max_norm_diff(&out, &out_twin);
    assert!(d < 5.0e-6, "extreme scores vs host twin: {d:e}");
}

/// Serving-length lanes, one length-1 lane beside them.
///
/// Two gaps in one shape. The grid's longest lane is 129 tokens, which is nowhere near a real
/// context and leaves the block walk's outer loop running a handful of iterations; these lanes run
/// it for thousands. And a 4096-token lane sharing a round with a 1-token lane is the round the
/// length gate exists for — if anything in the kernel reached for `blocks_per_lane` or `l_max`
/// instead of the lane's own length, the short lane would read the sentinel padding and the long
/// lane's blocks. Scored per lane, so the long lane's larger output cannot normalize that away.
#[test]
fn a_serving_length_lane_beside_a_length_one_lane() {
    for &(long, block_size) in &[(1024usize, 16usize), (4096, 128), (2049, 16), (2000, 3)] {
        check_per_lane(
            &Case {
                block_size,
                lengths: vec![1, long, 2, long - 1, 1],
                n_rep: 4,
                head_dim: 64,
                num_kv_heads: 8,
            },
            0x51DE + long as u64,
            &format!("long={long} bs={block_size}"),
        );
    }
}

/// Every length in a window straddling three block edges, one per lane, all in one round.
///
/// The kernel's per-block live count is `min(block_size, len - b·block_size)`. An off-by-one in it
/// is invisible for any length that happens to be a multiple of the block size and invisible again
/// for any round where every lane is the same length; it shows up here and essentially nowhere
/// else, because every residue is present at once and each lane is scored on its own.
#[test]
fn every_length_around_a_block_edge_agrees_per_lane() {
    for &block_size in &[1usize, 2, 5, 8, 16, 32] {
        check_per_lane(
            &Case {
                block_size,
                lengths: (1..=(3 * block_size + 2)).collect(),
                n_rep: 4,
                head_dim: 64,
                num_kv_heads: 8,
            },
            0xED9E + block_size as u64,
            &format!("block edge sweep bs={block_size}"),
        );
    }
}

/// A batch three times wider than the grid's widest.
///
/// The lane count is the cube grid's y extent — a runtime scalar, so this costs no extra shader
/// compilation — and 96 ragged lanes is the shape a server at width actually launches.
#[test]
fn a_batch_wider_than_the_grid_still_agrees() {
    check_per_lane(
        &Case {
            block_size: 32,
            lengths: (0..96).map(|j| 1 + (j * 13) % 200).collect(),
            n_rep: 4,
            head_dim: 64,
            num_kv_heads: 8,
        },
        0x5EED,
        "n=96",
    );
}

/// `head_dim` values that straddle the plane size rather than divide it.
///
/// The kernel slices a head across the plane cyclically and guards the ragged remainder. The grid
/// carries 64, 80 and 128; what it never asks is what happens when a head is *smaller* than a
/// plane (a lane of 17 elements on a 32-wide plane leaves most of it idle) or one element past a
/// plane boundary in either direction. Each value is a separate comptime specialization, so this
/// list is deliberately four entries and not twenty.
#[test]
fn head_dims_that_straddle_the_plane_size_agree() {
    for &head_dim in &[1usize, 17, 33, 65] {
        check_per_lane(
            &Case {
                block_size: 8,
                lengths: vec![1, 9, 16, 33],
                n_rep: 4,
                head_dim,
                num_kv_heads: 2,
            },
            0x4EAD + head_dim as u64,
            &format!("head_dim={head_dim}"),
        );
    }
}

/// Active lanes that are a non-contiguous subset of the pool, handed over out of order.
///
/// A plan's rows are `lanes` order, not lane-id order, and every index the kernel reads — the
/// block table, the lengths, the query rows — has to agree on that. A round built from lanes
/// `[0, 1, 2, ...]` cannot tell the difference between "row j" and "lane j"; this one can.
#[test]
fn an_out_of_order_lane_subset_agrees_with_the_reference() {
    let dev = f32_device();
    let (num_kv_heads, head_dim, n_rep, block_size) = (2usize, 64usize, 4usize, 8usize);
    let layout = KvLayout {
        n_layers: 1,
        n_kv_heads: num_kv_heads,
        head_dim,
        max_seq_len: 40,
    };
    let mut cache = PagedKvCache::with_window_per_lane(layout, 6, block_size, &dev);
    let mut noise = Noise::new(0xFEED);
    for (lane, prefill) in [(0usize, 30usize), (1, 3), (2, 17), (3, 9), (4, 25), (5, 1)] {
        let plan = cache.prepare_lanes(&[lane], prefill).unwrap();
        let shape = [1usize, num_kv_heads, prefill, head_dim];
        let kv = Tensor::<4>::from_data(
            TensorData::new(noise.vec(shape.iter().product()), shape),
            &dev,
        );
        for l in cache.layers_mut() {
            l.write(&plan, kv.clone(), kv.clone());
        }
    }

    let lanes = [4usize, 1, 5, 0];
    let n = lanes.len();
    let plan = cache.prepare_lanes(&lanes, 1).unwrap();
    let kv = Tensor::<4>::from_data(
        TensorData::new(
            noise.vec(n * num_kv_heads * head_dim),
            [n, num_kv_heads, 1, head_dim],
        ),
        &dev,
    );
    for l in cache.layers_mut() {
        l.write(&plan, kv.clone(), kv.clone());
    }
    let num_heads = num_kv_heads * n_rep;
    let q = Tensor::<4>::from_data(
        TensorData::new(
            noise.vec(n * num_heads * head_dim),
            [n, 1, num_heads, head_dim],
        ),
        &dev,
    )
    .swap_dims(1, 2);

    force_mode(PagedAttentionMode::Kernel);
    let before = kernel_launches();
    let out_kernel = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention(q.clone(), l, &plan, n_rep))
    };
    assert!(kernel_launches() > before, "the kernel never ran");
    let out_ref = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention_reference(q, l, &plan, n_rep))
    };
    let (lane, d) = worst_lane_diff(&out_kernel, &out_ref, n);
    assert!(d < 1.0e-5, "lane subset {lanes:?}: row {lane} -> {d:e}");
}

/// The GQA head mapping, as a permutation test.
///
/// A wrong `head = kv_head · n_rep + g` is invisible whenever the queries inside a group are
/// similar, because every head of the group then produces nearly the same answer. So this makes
/// exactly one head per group loud and the rest near-zero: the group's outputs spread far apart,
/// and any permutation of them is a large error rather than rounding. The spread is asserted too —
/// a version of this test where the heads happened to agree would prove nothing, and would look
/// exactly as green.
#[test]
fn the_gqa_head_mapping_is_not_a_permutation() {
    let dev = f32_device();
    let (num_kv_heads, n_rep, head_dim, block_size) = (3usize, 4usize, 64usize, 8usize);
    let num_heads = num_kv_heads * n_rep;
    let lengths = [17usize, 5, 33];
    let n = lengths.len();
    let layout = KvLayout {
        n_layers: 1,
        n_kv_heads: num_kv_heads,
        head_dim,
        max_seq_len: 33,
    };
    let mut cache = PagedKvCache::with_window_per_lane(layout, n, block_size, &dev);
    let mut noise = Noise::new(0x9A9A);
    for (lane, &len) in lengths.iter().enumerate() {
        let plan = cache.prepare_lanes(&[lane], len - 1).unwrap();
        let shape = [1usize, num_kv_heads, len - 1, head_dim];
        let count: usize = shape.iter().product();
        let k = Tensor::<4>::from_data(TensorData::new(noise.vec(count), shape), &dev);
        let v = Tensor::<4>::from_data(TensorData::new(noise.vec(count), shape), &dev);
        for l in cache.layers_mut() {
            l.write(&plan, k.clone(), v.clone());
        }
    }
    let plan = cache.prepare_lanes(&[0, 1, 2], 1).unwrap();
    let kv_shape = [n, num_kv_heads, 1, head_dim];
    let count: usize = kv_shape.iter().product();
    let k = Tensor::<4>::from_data(TensorData::new(noise.vec(count), kv_shape), &dev);
    let v = Tensor::<4>::from_data(TensorData::new(noise.vec(count), kv_shape), &dev);
    for l in cache.layers_mut() {
        l.write(&plan, k.clone(), v.clone());
    }

    let mut q_data = vec![0.0f32; n * num_heads * head_dim];
    for lane in 0..n {
        for head in 0..num_heads {
            // A different member of each group is the loud one, so a rotation within a group is
            // just as visible as a swap between groups.
            let loud = head % n_rep == (head / n_rep) % n_rep;
            for d in 0..head_dim {
                q_data[(lane * num_heads + head) * head_dim + d] =
                    if loud { noise.next() * 12.0 } else { 1.0e-4 };
            }
        }
    }
    // Built directly in `[n, heads, 1, d]` order rather than transposed into it: everywhere else
    // in this file `q` reaches the kernel as a strided view, so a bug that only bites a genuinely
    // contiguous query would have nowhere to show.
    let q = Tensor::<4>::from_data(TensorData::new(q_data, [n, num_heads, 1, head_dim]), &dev);

    force_mode(PagedAttentionMode::Kernel);
    let before = kernel_launches();
    let out_kernel = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention(q.clone(), l, &plan, n_rep))
    };
    assert!(kernel_launches() > before, "the kernel never ran");
    let out_ref = {
        let l = cache.layers_mut().next().unwrap();
        host(paged_attention_reference(q, l, &plan, n_rep))
    };

    // Scored per head slot, because a permutation is exactly what this is looking for.
    let (slot, worst) = worst_lane_diff(&out_kernel, &out_ref, n * num_heads);
    assert!(worst < 1.0e-4, "head slot {slot}: {worst:e}");

    let mut group_spread = 0.0f32;
    for lane in 0..n {
        for kv_head in 0..num_kv_heads {
            let first = ((lane * num_heads) + kv_head * n_rep) * head_dim;
            for g in 1..n_rep {
                let other = ((lane * num_heads) + kv_head * n_rep + g) * head_dim;
                group_spread = group_spread.max(max_norm_diff(
                    &out_ref[first..first + head_dim],
                    &out_ref[other..other + head_dim],
                ));
            }
        }
    }
    println!("gqa permutation test: worst {worst:e}, in-group spread {group_spread:e}");
    assert!(
        group_spread > 0.05,
        "the heads inside a group answer too similarly ({group_spread:e}): a permutation bug \
         would not be visible, so this test would prove nothing"
    );
}

/// The plan's three descriptions of where each lane stops must say the same thing.
///
/// `mask`, `lengths` and the sentinel-padded `gather_idx` are built in the same loop from the same
/// snapshot, and the reference reads the first while the kernel reads the second — so a
/// differential test that feeds both implementations the *same* plan stays green even if the two
/// disagree, and the disagreement only ever surfaces as one path reading another sequence's KV.
/// This checks them against each other directly, on the host, with no attention in the way.
#[test]
fn a_plans_mask_lengths_and_block_table_agree() {
    let dev = f32_device();
    let block_size = 8usize;
    let lengths = [1usize, 8, 9, 23, 40];
    let n = lengths.len();
    let layout = KvLayout {
        n_layers: 1,
        n_kv_heads: 1,
        head_dim: 8,
        max_seq_len: 40,
    };
    let mut cache = PagedKvCache::with_window_per_lane(layout, n, block_size, &dev);
    for (lane, &len) in lengths.iter().enumerate() {
        if len > 1 {
            cache.prepare_lanes(&[lane], len - 1).unwrap();
        }
    }
    let plan = cache.prepare_lanes(&(0..n).collect::<Vec<_>>(), 1).unwrap();

    let lens = host_ints(plan.lengths.clone());
    let mask: Vec<bool> = plan.mask.clone().into_data().iter::<bool>().collect();
    for (lane, &len) in lens.iter().enumerate() {
        for column in 0..plan.l_max {
            assert_eq!(
                mask[lane * plan.l_max + column],
                column >= len as usize,
                "lane {lane}: mask column {column} disagrees with length {len}"
            );
        }
    }

    let table = host_ints(plan.gather_idx.clone());
    for (lane, &len) in lens.iter().enumerate() {
        let live = (len as usize).div_ceil(block_size);
        for entry in live..plan.blocks_per_lane {
            assert_eq!(
                table[lane * plan.blocks_per_lane + entry],
                0,
                "lane {lane}: block table entry {entry} past the live prefix is not the sentinel"
            );
        }
    }
}

/// The roofline bench: what the kernel actually costs against what its reads cost.
///
/// Attention only — no model, no other layers — at the Llama-3.2-1B decode geometry (8 KV heads,
/// 32 query heads, head_dim 64, 128-token blocks) across the three context regimes the design
/// document names. Both implementations are timed the same way, and the sync is honest: each round
/// reads its output back to the host, so the number includes the whole round and cannot be a
/// queue-depth artifact.
///
/// The floor it is measured against is the one thing the kernel claims: it reads K and V exactly
/// once, so a round moves `2 · Σ_j ceil(len_j/bs)·bs · kv_heads · head_dim · 4` bytes and nothing
/// else. The reference's own traffic is several times that (gather, transpose, 4x head expansion,
/// then the attention op's own reads), which is what the ratio between the two columns is showing.
///
/// ```text
/// cargo test -p burn-lm-paged-kv --release --features metal \
///     kernel_tests::bench -- --ignored --nocapture
/// ```
#[test]
#[ignore = "benchmark: run manually, in release"]
fn bench_paged_decode_against_the_reference() {
    let dev = f32_device();
    let (num_kv_heads, n_rep, head_dim, block_size) = (8usize, 4usize, 64usize, 128usize);
    let num_heads = num_kv_heads * n_rep;
    let rounds = 10;
    let warmup = 3;
    // Launches enqueued per host sync. A single-round readback on this geometry is dominated by
    // submit + readback latency (several milliseconds of it), which hides whatever the kernel
    // itself costs; batching the launches behind one sync amortizes that to where the number
    // being reported is the kernel's.
    let inner = 20;

    println!(
        "paged decode vs reference, f32, kv_heads {num_kv_heads} x n_rep {n_rep}, head_dim \
         {head_dim}, block_size {block_size}, {rounds} timed rounds"
    );
    println!(
        "{:>6} {:>4} {:>12} {:>12} {:>9} {:>12}",
        "l_max", "n", "kernel ms", "ref ms", "speedup", "kernel GB/s"
    );

    for &l_max in &[128usize, 1024, 4096] {
        for &n in &[1usize, 8, 32] {
            let layout = KvLayout {
                n_layers: 1,
                n_kv_heads: num_kv_heads,
                head_dim,
                max_seq_len: l_max,
            };
            let mut cache = PagedKvCache::with_window_per_lane(layout, n, block_size, &dev);
            let mut noise = Noise::new(0xB0A7 + l_max as u64 * 31 + n as u64);

            // Every lane to l_max - 1, then one decode token each. Same length in every lane: this
            // is the bandwidth question, not the raggedness question.
            let prefill = l_max - 1;
            let kv_shape = [1usize, num_kv_heads, prefill, head_dim];
            let kv = Tensor::<4>::from_data(
                TensorData::new(noise.vec(kv_shape.iter().product()), kv_shape),
                &dev,
            );
            for lane in 0..n {
                let plan = cache.prepare_lanes(&[lane], prefill).unwrap();
                for l in cache.layers_mut() {
                    l.write(&plan, kv.clone(), kv.clone());
                }
            }
            let lanes: Vec<usize> = (0..n).collect();
            let plan = cache.prepare_lanes(&lanes, 1).unwrap();
            let step_shape = [n, num_kv_heads, 1, head_dim];
            let step = Tensor::<4>::from_data(
                TensorData::new(noise.vec(step_shape.iter().product()), step_shape),
                &dev,
            );
            for l in cache.layers_mut() {
                l.write(&plan, step.clone(), step.clone());
            }
            let q = Tensor::<4>::from_data(
                TensorData::new(
                    noise.vec(n * num_heads * head_dim),
                    [n, 1, num_heads, head_dim],
                ),
                &dev,
            )
            .swap_dims(1, 2);

            let mut time = |mode: PagedAttentionMode| {
                force_mode(mode);
                let mut start = std::time::Instant::now();
                for round in 0..rounds + warmup {
                    if round == warmup {
                        // Untimed warmup absorbs shader compilation and autotune.
                        start = std::time::Instant::now();
                    }
                    let l = cache.layers_mut().next().unwrap();
                    let mut last = None;
                    for _ in 0..inner {
                        last = Some(paged_attention(q.clone(), l, &plan, n_rep));
                    }
                    // The readback is the sync: without it this would time enqueueing.
                    let _ = host(last.unwrap());
                }
                start.elapsed().as_secs_f64() * 1e3 / (rounds * inner) as f64
            };

            let launches_before = kernel_launches();
            let kernel_ms = time(PagedAttentionMode::Kernel);
            assert!(
                kernel_launches() > launches_before,
                "l_max {l_max}, n {n}: the kernel never ran"
            );
            let ref_ms = time(PagedAttentionMode::Reference);

            // Every lane holds l_max positions, so the walk covers ceil(l_max/bs) whole blocks.
            let bytes = 2.0
                * (n * l_max.div_ceil(block_size) * block_size * num_kv_heads * head_dim * 4)
                    as f64;
            let gbps = bytes / (kernel_ms * 1e-3) / 1e9;
            println!(
                "{l_max:>6} {n:>4} {kernel_ms:>12.3} {ref_ms:>12.3} {:>8.2}x {gbps:>12.1}",
                ref_ms / kernel_ms
            );
        }
    }
    force_mode(PagedAttentionMode::Reference);
}
