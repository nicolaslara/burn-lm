//! A self-contained proof, runnable anywhere the kernel is built, that the device ran the kernel.
//!
//! The launch counter is necessary and not sufficient. It is incremented by our own host code one
//! line after the launch, so on its own it says "this code path was taken" — not "the GPU executed
//! the kernel body", and certainly not "the numbers are right". This check closes both gaps
//! without a rebuild, on whatever device it is handed:
//!
//! 1. the counter *discriminates*: a forced-kernel decode moves it, a forced-reference decode on
//!    the same inputs does not;
//! 2. the kernel's numbers agree with [`decode_reference_online`], a plain-Rust twin that never
//!    touched the GPU;
//! 3. a deliberate error injected into the kernel's *epilogue* — a runtime scalar folded into the
//!    final division inside the compiled kernel source, and nowhere else — reaches the output, and
//!    restoring it restores the output exactly.
//!
//! (3) is the part that cannot be faked by a fallback: if the reference path had produced these
//! numbers, the injected scale would have gone nowhere.
//!
//! It is deliberately tiny (two lanes, one round) so that a server can run it at startup and pay
//! milliseconds for it.

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::oracle::{decode_reference_online, DecodeShape};
use crate::switch::{force_mode, kernel_launches, set_output_scale, PagedAttentionMode};

/// What [`dispatch_self_check`] found. Every field is evidence; the interpretation is the
/// caller's, so that a server can log it and a test can assert on it.
#[derive(Debug, Clone, Copy)]
pub struct SelfCheck {
    /// How far the launch counter moved during the forced-kernel decode. Must be >= 1.
    pub kernel_launch_delta: usize,
    /// How far it moved during the forced-reference decode. Must be 0.
    pub reference_launch_delta: usize,
    /// Did the forced-kernel call actually return a kernel result, or decline?
    pub kernel_ran: bool,
    /// Largest absolute difference between the kernel and the plain-Rust twin.
    pub kernel_vs_oracle_max_abs: f32,
    /// The error that was injected: the epilogue scale, where `1.0` is the real kernel.
    pub injected_scale: f32,
    /// Largest absolute difference the injected epilogue error produced. Must be well above zero.
    pub injected_max_abs: f32,
    /// Largest absolute difference left after restoring the scale. Must be zero.
    pub restored_max_abs: f32,
    /// The largest magnitude in the oracle output, so the two diffs above can be read as relative.
    pub oracle_max_abs: f32,
}

/// Run the check on `device`, injecting `injected_scale` (`1.001` is a 0.1% error) into the
/// epilogue. The injection is process-wide for its duration, so this wants the process to itself —
/// call it at startup, before serving.
///
/// Size the injection against the storage dtype: at f16 a 0.1% error is only a couple of ulp on a
/// value near 1, so a run that wants an unarguable answer should also try something like `1.01`.
pub fn dispatch_self_check(device: &Device, injected_scale: f32) -> SelfCheck {
    let shape = DecodeShape {
        n: 2,
        num_kv_heads: 2,
        n_rep: 2,
        head_dim: 64,
        block_size: 4,
        blocks_per_lane: 2,
        scale: 0.125,
    };
    let num_heads = shape.num_kv_heads * shape.n_rep;
    let num_blocks = 4;

    // A cheap deterministic spread, in [-1, 1); nothing here needs a real RNG.
    let noise = |seed: u64, count: usize| -> Vec<f32> {
        let mut x = seed | 1;
        (0..count)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    };

    let q_host = noise(0xA11CE, shape.n * num_heads * shape.head_dim);
    let pool_len = num_blocks * shape.block_size * shape.num_kv_heads * shape.head_dim;
    let k_host = noise(0xB0B, pool_len);
    let v_host = noise(0xC0FFEE, pool_len);
    // Not the identity: a lane whose blocks are out of order is the mapping a bug gets right by
    // accident when the table is 0,1,2,...
    let table_host: Vec<i32> = vec![2, 0, 3, 1];
    let lengths_host: Vec<i32> = vec![7, 3];

    let pool_dims = [
        num_blocks,
        shape.block_size,
        shape.num_kv_heads,
        shape.head_dim,
    ];
    let build = || {
        let q = Tensor::<4>::from_data(
            TensorData::new(q_host.clone(), [shape.n, num_heads, 1, shape.head_dim]),
            device,
        );
        let k = Tensor::<4>::from_data(TensorData::new(k_host.clone(), pool_dims), device);
        let v = Tensor::<4>::from_data(TensorData::new(v_host.clone(), pool_dims), device);
        let table = Tensor::<1, Int>::from_data(
            TensorData::new(table_host.clone(), [table_host.len()]),
            device,
        );
        let lengths = Tensor::<1, Int>::from_data(
            TensorData::new(lengths_host.clone(), [lengths_host.len()]),
            device,
        );
        (q, k, v, table, lengths)
    };
    let run = || {
        let (q, k, v, table, lengths) = build();
        crate::paged_decode(q, k, v, table, lengths, shape.n_rep, shape.scale)
            .map(|out| out.into_data().iter::<f32>().collect::<Vec<f32>>())
    };
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    // 1. The counter, both ways.
    force_mode(PagedAttentionMode::Kernel);
    let before = kernel_launches();
    let kernel = run();
    let kernel_launch_delta = kernel_launches() - before;

    force_mode(PagedAttentionMode::Reference);
    let before = kernel_launches();
    let declined = run();
    let reference_launch_delta = kernel_launches() - before;
    debug_assert!(
        declined.is_none(),
        "forced reference returned a kernel result"
    );

    // 2 and 3 need the kernel; if it declined this device there is nothing left to compare.
    force_mode(PagedAttentionMode::Kernel);
    let Some(kernel) = kernel else {
        force_mode(PagedAttentionMode::Auto);
        return SelfCheck {
            kernel_launch_delta,
            reference_launch_delta,
            kernel_ran: false,
            injected_scale,
            kernel_vs_oracle_max_abs: f32::NAN,
            injected_max_abs: f32::NAN,
            restored_max_abs: f32::NAN,
            oracle_max_abs: f32::NAN,
        };
    };

    let oracle =
        decode_reference_online(&q_host, &k_host, &v_host, &table_host, &lengths_host, shape);
    let oracle_max_abs = oracle.iter().fold(0.0f32, |m, x| m.max(x.abs()));

    set_output_scale(injected_scale);
    let injected = run().expect("the kernel ran a moment ago");
    set_output_scale(1.0);
    let restored = run().expect("the kernel ran a moment ago");

    force_mode(PagedAttentionMode::Auto);
    SelfCheck {
        kernel_launch_delta,
        reference_launch_delta,
        kernel_ran: true,
        injected_scale,
        kernel_vs_oracle_max_abs: worst(&kernel, &oracle),
        injected_max_abs: worst(&kernel, &injected),
        restored_max_abs: worst(&kernel, &restored),
        oracle_max_abs,
    }
}
