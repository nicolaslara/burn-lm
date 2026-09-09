//! The opt-in switch, and the counter that keeps a silent fallback from being invisible.

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Which paged-attention implementation the process runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedAttentionMode {
    /// Generic tensor ops: gather, transpose, expand, fused attention under the plan's mask. The
    /// default, and the oracle every kernel test compares against.
    Reference,
    /// The paged decode kernel, where it applies; the reference everywhere else.
    Kernel,
}

/// `BURN_LM_PAGED_ATTENTION` — `kernel` or `reference`, read once.
///
/// The default is `reference` on purpose. `OperationIr::Custom` is a fusion barrier, so the kernel
/// costs whatever elementwise fusion currently wraps the attention chain; at short context that can
/// be more than the copies it removes. Until there is a number saying otherwise, the kernel is
/// something you ask for.
fn parse_mode() -> PagedAttentionMode {
    match std::env::var("BURN_LM_PAGED_ATTENTION").as_deref() {
        Ok("kernel") => {
            log::info!("burn-lm paged decode: BURN_LM_PAGED_ATTENTION=kernel");
            PagedAttentionMode::Kernel
        }
        Ok("reference") | Err(_) => {
            log::info!("burn-lm paged decode: mode=reference (tensor-op path)");
            PagedAttentionMode::Reference
        }
        Ok(other) => {
            log::warn!(
                "BURN_LM_PAGED_ATTENTION={other:?} is not one of `kernel` / `reference`; \
                 using `reference`"
            );
            PagedAttentionMode::Reference
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
        log::info!(
            "burn-lm paged decode: kernel launches = {}",
            previous + 1
        );
    }
}

/// How many times `paged_decode` has taken the kernel path in this process.
///
/// The fallback is silent by design, and that cuts both ways: a test that only checks output
/// equivalence passes just as happily when the kernel never ran. Assert this moved.
pub fn kernel_launches() -> usize {
    LAUNCHES.load(Ordering::Relaxed)
}
