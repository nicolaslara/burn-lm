//! The dispatch self-check, run where it can own the process.
//!
//! It injects a process-wide scale into the kernel epilogue, so it cannot share a test binary with
//! the differential grid in `burn-lm-paged-kv` — hence its own integration target. The same
//! function runs at server startup on a GPU we cannot run `cargo test` on, which is the whole
//! reason it is library code rather than a test.

#![cfg(feature = "kernel")]

use burn_lm_paged_attn::dispatch_self_check;

#[test]
fn the_device_runs_the_kernel_body() {
    let device = burn::tensor::Device::default();
    let check = dispatch_self_check(&device, 1.001);
    println!("{check:#?}");

    assert!(
        check.kernel_ran,
        "this device declined the kernel: {check:?}"
    );
    assert!(
        check.kernel_launch_delta >= 1,
        "a forced-kernel decode did not move the launch counter: {check:?}"
    );
    assert_eq!(
        check.reference_launch_delta, 0,
        "a forced-reference decode moved the launch counter: {check:?}"
    );
    assert!(
        check.kernel_vs_oracle_max_abs < 1.0e-5,
        "the kernel disagrees with the plain-Rust twin: {check:?}"
    );
    assert!(
        check.injected_max_abs > 1.0e-5,
        "a 0.1% error injected into the kernel epilogue never reached the output — these numbers \
         did not come out of the kernel: {check:?}"
    );
    assert_eq!(
        check.restored_max_abs, 0.0,
        "restoring the epilogue scale did not restore the output: {check:?}"
    );
}
