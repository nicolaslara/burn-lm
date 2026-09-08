//! The opt-in switch, and the counter that keeps a silent fallback from being invisible.

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
        Ok("kernel") => PagedAttentionMode::Kernel,
        Ok("reference") | Err(_) => PagedAttentionMode::Reference,
        Ok(other) => {
            log::warn!(
                "BURN_LM_PAGED_ATTENTION={other:?} is not one of `kernel` / `reference`; \
                 using `reference`"
            );
            PagedAttentionMode::Reference
        }
    }
}

pub(crate) fn mode() -> PagedAttentionMode {
    static MODE: OnceLock<PagedAttentionMode> = OnceLock::new();
    *MODE.get_or_init(parse_mode)
}

static LAUNCHES: AtomicUsize = AtomicUsize::new(0);

#[cfg_attr(not(feature = "kernel"), allow(dead_code))]
pub(crate) fn count_launch() {
    LAUNCHES.fetch_add(1, Ordering::Relaxed);
}

/// How many times `paged_decode` has taken the kernel path in this process.
///
/// The fallback is silent by design, and that cuts both ways: a test that only checks output
/// equivalence passes just as happily when the kernel never ran. Assert this moved.
pub fn kernel_launches() -> usize {
    LAUNCHES.load(Ordering::Relaxed)
}
