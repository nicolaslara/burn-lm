//! Rewrite a Llama checkpoint from the legacy `.mpk` msgpack format into burn's current burnpack
//! format, once, so later loads take the fast path.
//!
//! The published checkpoints (see `pretrained.rs`) are all `.mpk`, the format burn wrote before the
//! record refactor. We can still read them — `nn/llama/legacy_mpk.rs` is a streaming msgpack
//! visitor written for exactly that — but reading one costs several seconds every load, because
//! the whole document has to be walked and every tensor converted on the way in. A burnpack file
//! of the same weights loads in a fraction of that time. This example is the one-shot rewrite:
//! `Llama::load` reads whatever format the input is in, `Llama::save` writes burnpack.
//!
//! ```text
//! cargo run --release --example convert --features llama3,test-metal -- --model llama3.2-1b
//! ```
//!
//! With no `--input` it converts the cached pretrained weights for `--model`, and with no
//! `--output` it writes a `.bpk` next to the input — which is where the loader looks for it
//! (`Pretrained::checkpoint`), so a bare run migrates the local cache. Note that the model still
//! has to be built to be loaded into, so `--model` has to name the weights the input holds; a
//! mismatch is reported by the load rather than silently written out.

use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::Device;
use burn_lm_llama::{
    inference::Llama, pretrained::ModelMeta, tokenizer::Tokenizer, LlamaConfig, LlamaVersion,
    TinyLlamaVersion,
};
use clap::{Parser, ValueEnum};

/// The KV window to build the model with. Converting never runs a forward pass, so the cache is
/// dead weight here; keep it small so the process pays for weights only.
const MAX_SEQ_LEN: usize = 512;

#[derive(Parser, Debug)]
#[command(about = "Convert a Llama checkpoint from the legacy .mpk format to burnpack")]
struct Args {
    /// Which weights the input holds. Also picks the cached checkpoint used when `--input` is
    /// absent.
    #[arg(short, long, value_enum)]
    model: Model,

    /// The checkpoint to read. Defaults to the cached pretrained weights for `--model`.
    #[arg(short, long)]
    input: Option<String>,

    /// Where to write the burnpack record. Defaults to the input path with a `.bpk` extension.
    #[arg(short, long)]
    output: Option<String>,
}

/// The models whose weights this example can build a module for. One per config in `LlamaConfig`
/// that has a published checkpoint behind it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Model {
    #[value(name = "llama3.2-1b")]
    Llama32_1b,
    #[value(name = "llama3.2-3b")]
    Llama32_3b,
    #[value(name = "tinyllama")]
    TinyLlama,
}

fn main() {
    let args = Args::parse();
    let device = Device::default();
    println!("device: {device:?}");

    // Weights and tokenizer from the local cache, downloaded on first use — the same source
    // `quantize.rs` reads.
    let (cached_checkpoint, tokenizer) = match args.model {
        Model::Llama32_1b => cached(LlamaVersion::Llama321bInstruct),
        Model::Llama32_3b => cached(LlamaVersion::Llama323bInstruct),
        Model::TinyLlama => cached(TinyLlamaVersion::V1),
    };

    let input = args.input.map(PathBuf::from).unwrap_or(cached_checkpoint);
    let output = match args.output {
        Some(path) => PathBuf::from(path),
        None => input.with_extension("bpk"),
    };
    let input_bytes = file_size(&input);
    println!("input:  {} ({})", input.display(), human_size(input_bytes));

    let now = Instant::now();
    let checkpoint = input.to_str().expect("input path should be valid UTF-8");
    let tokenizer = tokenizer.to_str().expect("tokenizer path should be UTF-8");
    // Each config builds a different `Llama<T>`, so the load has to branch here; everything after
    // it is the same for all of them, hence the shared `save`.
    let elapsed = match args.model {
        Model::Llama32_1b => {
            let llama =
                LlamaConfig::load_llama3_2_1b(checkpoint, tokenizer, MAX_SEQ_LEN, 1, 0, &device)
                    .expect("should load the checkpoint");
            report_load(now);
            save(llama, &output)
        }
        Model::Llama32_3b => {
            let llama =
                LlamaConfig::load_llama3_2_3b(checkpoint, tokenizer, MAX_SEQ_LEN, 1, 0, &device)
                    .expect("should load the checkpoint");
            report_load(now);
            save(llama, &output)
        }
        Model::TinyLlama => {
            let llama = load_tiny_llama(checkpoint, tokenizer, &device);
            report_load(now);
            save(llama, &output)
        }
    };

    let output_bytes = file_size(&output);
    println!(
        "output: {} ({}) written in {:.1}s",
        output.display(),
        human_size(output_bytes),
        elapsed
    );
}

/// TinyLlama needs the SentencePiece tokenizer, so it is only loadable when the crate was built
/// with the `tiny` feature. The example itself only requires `llama3` (the other two models), so
/// asking for TinyLlama without that feature is a run-time refusal rather than a build failure.
#[cfg(feature = "tiny")]
fn load_tiny_llama(
    checkpoint: &str,
    tokenizer: &str,
    device: &Device,
) -> Llama<burn_lm_llama::tokenizer::SentencePieceTokenizer> {
    LlamaConfig::load_tiny_llama(checkpoint, tokenizer, MAX_SEQ_LEN, device)
        .expect("should load the checkpoint")
}

#[cfg(not(feature = "tiny"))]
fn load_tiny_llama(_: &str, _: &str, _: &Device) -> Llama<burn_lm_llama::tokenizer::Tiktoken> {
    panic!("TinyLlama needs its SentencePiece tokenizer: rebuild with `--features tiny`")
}

/// Write the loaded weights out as burnpack, returning how long it took.
fn save<T: Tokenizer>(llama: Llama<T>, output: &Path) -> f64 {
    let now = Instant::now();
    llama
        .save(output.to_str().expect("output path should be valid UTF-8"))
        .expect("should save the record");
    now.elapsed().as_secs_f64()
}

fn report_load(now: Instant) {
    println!("loaded in {:.1}s", now.elapsed().as_secs_f64());
}

/// The cached weights and tokenizer for one published model.
fn cached<M: ModelMeta>(version: M) -> (PathBuf, PathBuf) {
    let pretrained = version.pretrained();
    let checkpoint = pretrained
        .download_weights()
        .expect("should have the weights available");
    let tokenizer = pretrained
        .download_tokenizer()
        .expect("should have the tokenizer available");
    (checkpoint, tokenizer)
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path)
        .map(|meta| meta.len())
        .unwrap_or_else(|err| panic!("should stat {}: {err}", path.display()))
}

fn human_size(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}
