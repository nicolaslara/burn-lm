//! A paged KV cache for batched, decoder-only models.
//!
//! Under a per-lane slab, KV capacity is the rectangle `max_lanes × max_seq_len`, paid in full
//! whether a sequence is a 200-token chat turn or an 8000-token document — width and context are
//! zero-sum. This crate cuts KV storage into fixed-size blocks allocated on demand: a lane holds
//! only the blocks covering the tokens it has actually written, and capacity becomes the sum of
//! what live sequences actually need.
//!
//! The pieces, bottom up:
//!
//! - [`BlockPool`] — the per-lane ledger: each lane's length and block table, plus the shared free
//!   stack. `begin_round` grows tables and advances lengths together, all lanes or none.
//! - [`BlockStore`] — the physical bytes: one tensor of blocks, written by table+offset, read by
//!   one indexed gather. Knows nothing about lanes.
//! - [`KeyValueCache`] — a K and a V store side by side; one block id addresses both.
//! - [`LanePlan`] — one round's complete addressing contract: positions, tables, the prebuilt
//!   gather index, and the per-lane causal+padding mask.
//! - [`PagedKvCache`] — the model-facing cache: per-layer stores plus the ledger, planning one
//!   round at a time via `prepare_lanes`.
//!
//! A model opts in by owning a [`PagedKvCache`] and threading each round's [`LanePlan`] down to its
//! attention layers — see `burn-lm-llama` for the reference integration. Opting *out* of paging is
//! not a different type: [`PagedKvCache::unpaged`] spans each lane's whole context with one block,
//! which is exactly the old slab layout (and is covered by the same equivalence gates).
//!
//! This crate is model-side machinery. It deliberately has no dependency on the serving engine
//! (`burn-lm-inference`), and nothing here crosses the engine's decoder trait — the engine speaks
//! slots and token ids; blocks stay a private concern of the model that chose them.

mod attention;
mod block_pool;
mod block_store;
mod cache;
mod kv_cache;

// One accessor for every test device in this crate, so the second one (the f16 grid's) is brought
// up before anything is running on the first. See the module docs for why that matters.
#[cfg(test)]
mod test_device;

pub use attention::{paged_attention, paged_attention_reference};
pub use block_pool::*;

/// The decode kernel's switch and launch counter, re-exported so a model that owns a
/// [`PagedKvCache`] can drive both implementations without depending on the kernel crate directly.
/// `burn-lm-llama`'s model-level gate is the caller: it runs the same decoder twice, once each way,
/// and asserts the kernel actually ran — a silent fallback is by design, and that cuts both ways.
pub use burn_lm_paged_attn::{kernel_launches, PagedAttentionMode};
pub use cache::*;
pub use kv_cache::*;
