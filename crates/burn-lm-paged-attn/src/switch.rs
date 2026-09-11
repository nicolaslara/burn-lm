//! The opt-in switch, and the counter that keeps a silent fallback from being invisible.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Which paged-attention implementation the process runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedAttentionMode {
    /// Generic tensor ops: gather, transpose, expand, fused attention under the plan's mask. The
    /// oracle every kernel test compares against, and the implementation for everything the
    /// kernel does not serve.
    Reference,
    /// The paged decode kernel, where it applies; the reference everywhere else.
    Kernel,
    /// Let the device decide. The default, and the only mode whose answer depends on where the
    /// tensors live — see `crate::backend::device_defaults_to_kernel`.
    Auto,
}

/// `BURN_LM_PAGED_ATTENTION` — `kernel`, `reference` or `auto`, read once.
///
/// The default is `auto`, which asks the device. It used to be `reference` for a good reason:
/// `OperationIr::Custom` is a fusion barrier, so the kernel costs whatever elementwise fusion
/// currently wraps the attention chain, and at short context that could be more than the copies it
/// removes. The measurements are now in, and they are not the same on every backend, so the
/// default is per-device rather than one answer for all of them. `kernel` and `reference` are
/// still absolute: they say run it, or do not, wherever the tensors are.
fn parse_mode() -> PagedAttentionMode {
    match std::env::var("BURN_LM_PAGED_ATTENTION").as_deref() {
        Ok("kernel") => {
            tracing::info!("burn-lm paged decode: BURN_LM_PAGED_ATTENTION=kernel");
            PagedAttentionMode::Kernel
        }
        Ok("reference") => {
            tracing::info!("burn-lm paged decode: BURN_LM_PAGED_ATTENTION=reference");
            PagedAttentionMode::Reference
        }
        Ok("auto") | Err(_) => PagedAttentionMode::Auto,
        Ok(other) => {
            tracing::warn!(
                "BURN_LM_PAGED_ATTENTION={other:?} is not one of `kernel` / `reference` / \
                 `auto`; using `auto`"
            );
            PagedAttentionMode::Auto
        }
    }
}

thread_local! {
    /// This thread's override of the env switch, if it set one.
    static FORCED: Cell<Option<PagedAttentionMode>> = const { Cell::new(None) };
}

/// Force the mode on *this thread*, overriding `BURN_LM_PAGED_ATTENTION`.
///
/// For tests and benchmarks, which need both paths in one binary and cannot get that from an
/// environment variable read once at startup — and must not mutate the environment of a
/// multi-threaded process to try.
///
/// Per-thread rather than per-process, and that is the whole point. `paged_decode` reads the mode
/// on the thread that called it, so a test thread's choice reaches exactly its own rounds. A
/// process-wide switch would instead reach into whatever other tests happened to be running
/// alongside it — and the tests it would reach into are equivalence suites asserting *byte-exact*
/// argmax streams between two runs, which a mid-run implementation swap can flip on a near-tie.
/// That failure would be rare, load-dependent, and blamed on the kernel.
pub fn force_mode(mode: PagedAttentionMode) {
    FORCED.with(|forced| forced.set(Some(mode)));
}

pub(crate) fn mode() -> PagedAttentionMode {
    if let Some(mode) = FORCED.with(|forced| forced.get()) {
        return mode;
    }
    static MODE: OnceLock<PagedAttentionMode> = OnceLock::new();
    *MODE.get_or_init(parse_mode)
}

static LAUNCHES: AtomicUsize = AtomicUsize::new(0);

#[cfg_attr(not(feature = "kernel"), allow(dead_code))]
pub(crate) fn count_launch() {
    let previous = LAUNCHES.fetch_add(1, Ordering::Relaxed);
    // The first launch is the one that matters: it is the only positive evidence that every gate
    // between the env switch and the GPU said yes. After that, a decade-spaced heartbeat keeps the
    // count visible in a long benchmark without writing a line per decode round.
    if previous == 0 || (previous + 1) % 10_000 == 0 {
        tracing::info!("burn-lm paged decode: kernel launches = {}", previous + 1);
    }
}

/// How many times `paged_decode` has taken the kernel path in this process.
///
/// The fallback is silent by design, and that cuts both ways: a test that only checks output
/// equivalence passes just as happily when the kernel never ran. Assert this moved.
pub fn kernel_launches() -> usize {
    LAUNCHES.load(Ordering::Relaxed)
}

/// A deliberate error injected into the kernel's epilogue, for proving that the GPU ran *this*
/// kernel.
///
/// The launch counter says a host code path was taken; it cannot say the device executed the
/// kernel body, because our own host code is what increments it. This scalar is multiplied into
/// the kernel's final division, inside the compiled kernel source and nowhere else, so an output
/// that moves when it is set is direct evidence that the numbers came out of the kernel rather
/// than out of the fallback.
///
/// Process-wide, unlike `force_mode`, and not by preference: the launch happens on the dispatch
/// backend's worker thread, not the thread that called `paged_decode`, so a thread-local would
/// never be seen by the launch. A test that sets it therefore has to own the process — which is
/// why the one that does lives in its own integration-test binary.
///
/// Nothing in serving sets it, and at `1.0` it costs one multiply per output vector.
static OUTPUT_SCALE: AtomicU32 = AtomicU32::new(0x3f80_0000); // 1.0f32

/// Set the epilogue's injected output scale, process-wide. `1.0` is the real kernel.
pub fn set_output_scale(scale: f32) {
    OUTPUT_SCALE.store(scale.to_bits(), Ordering::Relaxed);
}

#[cfg_attr(not(feature = "kernel"), allow(dead_code))]
pub(crate) fn output_scale() -> f32 {
    f32::from_bits(OUTPUT_SCALE.load(Ordering::Relaxed))
}
