//! Loading a Llama straight out of a Hugging Face repo (the `hf` feature).
//!
//! Today's five models each need a checkpoint someone converted into a burn-only format and
//! published to a `tracel-ai/*-burn` repo, plus a hand-written [`LlamaConfig`] constructor stating
//! their dimensions. That is the single fact that makes the model list closed: a fine-tune of the
//! 1B you trained yourself cannot be run without both of those, and one of them you cannot do at
//! all.
//!
//! This path takes a repo id instead. It fetches the repo's own `config.json`, derives the module's
//! shape from it, fetches the canonical `model.safetensors` everyone else reads, and applies it
//! through `burn_store::SafetensorsStore`. No conversion, no publish, no per-model Rust.
//!
//! ```text
//! let llama = load_hf("unsloth/Llama-3.2-1B-Instruct", None, 2048, 1, 0, &device)?;
//! ```
//!
//! The published path is untouched: `pretrained.rs` and its six loaders still read their `.mpk`,
//! still cache burnpack beside it, and gain nothing and lose nothing from this.
//!
//! ## What this does not do yet
//!
//! **Sharded checkpoints.** Anything above roughly 7 B ships as `model-00001-of-000NN.safetensors`
//! plus a `model.safetensors.index.json` mapping each weight to its file, and burn-store has no
//! notion of either — every constructor takes one path. A repo laid out that way is refused with a
//! message saying so rather than half-loaded. Single-file repos, which is every 1B and 3B, work.
//!
//! **Other architectures.** The config's `architectures` is checked and anything but
//! `LlamaForCausalLM` is refused. Reading safetensors does not make our Llama into a Qwen; it makes
//! every Llama-shaped checkpoint on the Hub loadable, which is thousands of fine-tunes.
//!
//! **Chat templates.** A repo's `tokenizer_config.json` carries a Jinja `chat_template`, and we
//! still frame prompts with a hardcoded string. Rendering Jinja is a separate decision.

mod config;
mod repo;
mod weights;

pub use config::HfConfig;
pub use repo::HfRepo;
pub use weights::OutputWeights;

use std::path::Path;
use std::time::Instant;

use burn::prelude::*;

use crate::tokenizer::HfTokenizer;
use crate::{inference::Llama, LlamaConfig};

/// The file a single-file repo keeps its weights in.
const WEIGHTS: &str = "model.safetensors";
/// The manifest a sharded repo keeps instead, which is how sharding is detected.
const SHARD_INDEX: &str = "model.safetensors.index.json";
/// The tokenizer a canonical repo ships.
const TOKENIZER: &str = "tokenizer.json";
/// The name of the burnpack copy written beside the downloaded weights.
const CACHE: &str = "model.bpk";

/// Load a Llama from a Hugging Face repo id.
///
/// `revision` pins a branch, tag, or commit sha; `None` means the repo's default branch. The
/// remaining arguments size the module the same way the published loaders' do: `max_seq_len` is the
/// KV window, `max_batch_size` the number of concurrent lanes, and `kv_pool_tokens` the paged pool
/// (`0` for one full window per lane).
///
/// Files are cached under `~/.cache/llama/hf/<owner>__<name>/`, and the first load writes a
/// burnpack copy of the weights there so every later load reads that instead — the same trick the
/// published path uses for its `.mpk`, and worth rather more here, since it also saves redoing the
/// transpose, the dtype cast, and the rotary permutation.
pub fn load_hf(
    repo_id: &str,
    revision: Option<&str>,
    max_seq_len: usize,
    max_batch_size: usize,
    kv_pool_tokens: usize,
    device: &Device,
) -> Result<Llama<HfTokenizer>, String> {
    let mut repo = HfRepo::new(repo_id);
    if let Some(revision) = revision {
        repo = repo.with_revision(revision);
    }

    // The config comes first, and is cheap: it decides the module's shape, and refuses an
    // architecture or a RoPE scaling we cannot honestly build before any weights are fetched.
    let config_path = repo.file("config.json")?;
    let hf_config = HfConfig::read(&config_path)?;

    // Whether the weights are in one file, checked before the tokenizer is fetched: a sharded repo
    // is refused, and there is no reason to pull down a 17 MB tokenizer for a model we are about to
    // turn away.
    ensure_not_sharded(&repo)?;

    let tokenizer_path = repo.file(TOKENIZER)?;
    let tokenizer_path = path_str(&tokenizer_path)?;

    let llama_config = hf_config
        .to_llama_config(tokenizer_path)?
        .with_max_seq_len(max_seq_len)
        .with_max_batch_size(max_batch_size)
        .with_kv_pool_tokens(kv_pool_tokens);

    if let Some(max_position) = hf_config.max_position_embeddings {
        if max_seq_len > max_position {
            return Err(format!(
                "{repo_id} states a context length of {max_position} tokens; \
                 asked for {max_seq_len}"
            ));
        }
    }

    let mut llama = llama_config.init::<HfTokenizer>(device)?;

    // The burnpack copy of a previous load, if there is one.
    let cache = repo.cache_dir().join(CACHE);
    if cache.exists() {
        let now = Instant::now();
        let record = burn::store::ModuleRecord::load(&cache)
            .map_err(|err| format!("could not read {}: {err}", cache.display()))?;
        llama.decoder.model = llama
            .decoder
            .model
            .try_load_record(record)
            .map_err(|err| format!("failed to apply {}: {err}", cache.display()))?;
        println!(
            "Loaded {} from {} in {:.1}s",
            repo_id,
            cache.display(),
            now.elapsed().as_secs_f64()
        );
        return Ok(llama);
    }

    let checkpoint = repo.file(WEIGHTS)?;
    let now = Instant::now();
    let output = weights::load_safetensors(
        &mut llama.decoder.model,
        &checkpoint,
        device,
        hf_config.tie_word_embeddings,
        hf_config.num_attention_heads,
        hf_config.kv_heads(),
        hf_config.hidden_size,
    )?;
    println!(
        "Loaded {} ({} weights, output {}) in {:.1}s",
        repo_id,
        hf_config.torch_dtype.as_deref().unwrap_or("unstated"),
        match output {
            OutputWeights::FromCheckpoint => "from lm_head",
            OutputWeights::TiedToEmbedding => "tied to the embedding",
        },
        now.elapsed().as_secs_f64()
    );

    super::pretrained::write_burnpack(
        &llama.decoder.model,
        &cache,
        &checkpoint.display().to_string(),
    );

    Ok(llama)
}

/// Refuse a sharded repo by name, rather than letting it fail later as "no model.safetensors".
///
/// The shards are there; they are just not something burn-store can be pointed at yet. Saying so
/// is the difference between a known limit and a mystery.
fn ensure_not_sharded(repo: &HfRepo) -> Result<(), String> {
    match repo.optional_file(SHARD_INDEX)? {
        None => Ok(()),
        Some(index) => Err(format!(
            "{} is sharded ({} lists the shards); this loader reads a single \
             {WEIGHTS} only. burn-store has no shard-index support, so loading it means reading \
             the manifest, applying one store per shard partially, and checking the union covers \
             the module — a follow-up, not this change.",
            repo.repo_id(),
            index.display()
        )),
    }
}

fn path_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("{} is not valid UTF-8", path.display()))
}

impl LlamaConfig {
    /// Derive a config from a repo's `config.json`, without fetching anything else.
    ///
    /// Useful on its own for checking what a repo would build before committing to a download.
    pub fn from_hf_config(hf: &HfConfig, tokenizer_path: &str) -> Result<Self, String> {
        hf.to_llama_config(tokenizer_path)
    }
}
