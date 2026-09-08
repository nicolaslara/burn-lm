mod base;
pub use base::*;

#[cfg(feature = "pretrained")]
pub mod pretrained;

#[cfg(feature = "import")]
pub mod import;

/// Loading a model straight out of a Hugging Face repo: its own `config.json` and the canonical
/// safetensors weights, with no converted artifact in between.
#[cfg(feature = "hf")]
pub mod hf;

/// Reader for the msgpack weight files older burn versions wrote (the published checkpoints).
mod legacy_mpk;

pub mod inference;
pub mod training;
