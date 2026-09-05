//! Generate greedily from a Llama checkpoint, quantized or not, in either weight format.
//!
//! This is the check that goes with `examples/quantize.rs` and `examples/convert.rs`: point it at
//! the file one of those examples wrote and at the cached weights it came from in turn, give both
//! the same prompts, and the answers sit side by side. Sampling is the framework's `Argmax`, so a
//! run is deterministic and the only thing that differs between the two is the weights — which for
//! a format conversion is nothing at all, so the two transcripts have to match token for token.
//!
//! ```text
//! cargo run --release --example generate --features llama3,test-metal -- \
//!     --checkpoint /tmp/llama-3.2-1b-instruct-q4fb32.bpk \
//!     "What is the capital of France? One sentence."
//! ```
//!
//! With no `--checkpoint` it loads the pretrained weights for `--model` from the local cache, which
//! is the baseline half of that comparison.

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::Device;
use burn_lm_inference::{Argmax, GeneratedItemEmitter, TextGenerationListener};
use burn_lm_llama::{
    inference::Llama, pretrained::ModelMeta, tokenizer::Tokenizer, LlamaConfig, LlamaVersion,
    TinyLlamaVersion,
};
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(about = "Greedily generate from a Llama checkpoint")]
struct Args {
    /// Which model the checkpoint holds. Also picks the cached weights and tokenizer used when
    /// `--checkpoint` is absent.
    #[arg(short, long, value_enum, default_value_t = Model::Llama32_1b)]
    model: Model,

    /// A local record to load (burnpack, or a legacy `.mpk`). Defaults to the cached
    /// pretrained weights for `--model`.
    #[arg(short, long)]
    checkpoint: Option<String>,

    /// How many tokens to generate per prompt.
    #[arg(short, long, default_value_t = 64)]
    sample_len: usize,

    /// The KV window to size the model with.
    #[arg(long, default_value_t = 2048)]
    max_seq_len: usize,

    /// The prompts, each run as its own single-turn chat.
    prompts: Vec<String>,
}

/// The models this example can build a module for, matching `examples/convert.rs`.
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

    // The tokenizer always comes from the pretrained cache: it is the same file whatever the
    // weights were quantized to or saved as, and a local record carries weights only.
    let (cached_checkpoint, tokenizer) = match args.model {
        Model::Llama32_1b => cached(LlamaVersion::Llama321bInstruct),
        Model::Llama32_3b => cached(LlamaVersion::Llama323bInstruct),
        Model::TinyLlama => cached(TinyLlamaVersion::V1),
    };
    let checkpoint = args
        .checkpoint
        .clone()
        .map(PathBuf::from)
        .unwrap_or(cached_checkpoint);
    println!("checkpoint: {}", checkpoint.display());

    let now = Instant::now();
    let checkpoint = checkpoint
        .to_str()
        .expect("checkpoint path should be UTF-8");
    let tokenizer = tokenizer.to_str().expect("tokenizer path should be UTF-8");
    // Each config builds a different `Llama<T>`, so the load branches here and the generation loop
    // below is generic over the tokenizer.
    match args.model {
        Model::Llama32_1b => {
            let llama = LlamaConfig::load_llama3_2_1b(
                checkpoint,
                tokenizer,
                args.max_seq_len,
                1,
                0,
                &device,
            )
            .expect("should load the model");
            run(llama, now, &args);
        }
        Model::Llama32_3b => {
            let llama = LlamaConfig::load_llama3_2_3b(
                checkpoint,
                tokenizer,
                args.max_seq_len,
                1,
                0,
                &device,
            )
            .expect("should load the model");
            run(llama, now, &args);
        }
        Model::TinyLlama => {
            let llama = load_tiny_llama(checkpoint, tokenizer, args.max_seq_len, &device);
            run(llama, now, &args);
        }
    }
}

/// TinyLlama needs the SentencePiece tokenizer, so it is only loadable when the crate was built
/// with the `tiny` feature; asking for it without that feature is a run-time refusal rather than a
/// build failure for the other two models.
#[cfg(feature = "tiny")]
fn load_tiny_llama(
    checkpoint: &str,
    tokenizer: &str,
    max_seq_len: usize,
    device: &Device,
) -> Llama<burn_lm_llama::tokenizer::SentencePieceTokenizer> {
    LlamaConfig::load_tiny_llama(checkpoint, tokenizer, max_seq_len, device)
        .expect("should load the model")
}

#[cfg(not(feature = "tiny"))]
fn load_tiny_llama(
    _: &str,
    _: &str,
    _: usize,
    _: &Device,
) -> Llama<burn_lm_llama::tokenizer::Tiktoken> {
    panic!("TinyLlama needs its SentencePiece tokenizer: rebuild with `--features tiny`")
}

/// Run every prompt through the model, reporting the load time measured from `loading`.
fn run<T: Tokenizer + 'static>(mut llama: Llama<T>, loading: Instant, args: &Args) {
    println!("loaded in {:.1}s", loading.elapsed().as_secs_f64());

    for prompt in &args.prompts {
        let framed = frame(args.model, prompt);

        let (emitter, handle) = GeneratedItemEmitter::init(TextGenerationListener::default());
        let output = llama
            .generate(&framed, args.sample_len, &Argmax, emitter)
            .expect("should generate");
        let text = handle.join();

        println!("\n--- prompt: {prompt}");
        println!("{text}");
        println!(
            "--- {} tokens in {:.2}s ({:.2} tok/s)",
            output.tokens,
            output.time.as_secs_f64(),
            output.tokens as f64 / output.time.as_secs_f64()
        );
    }
}

/// Wrap a prompt in the single-turn chat template its model was instruction-tuned for — the same
/// framing the matching server builds (see `server/llama3.rs` and `server/tiny.rs`).
fn frame(model: Model, prompt: &str) -> String {
    match model {
        Model::Llama32_1b | Model::Llama32_3b => format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        Model::TinyLlama => format!("<|user|>\n{prompt}</s>\n<|assistant|>\n"),
    }
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
