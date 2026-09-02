mod base;
pub use base::*;

#[cfg(feature = "pretrained")]
pub mod pretrained;

#[cfg(feature = "import")]
pub mod import;

/// Reader for the msgpack weight files older burn versions wrote (the published checkpoints).
mod legacy_mpk;

pub mod inference;
pub mod training;
