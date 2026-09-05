//! Generate greedily from a Llama-3.2-1B-Instruct checkpoint, quantized or not.
//!
//! This is the check that goes with `examples/quantize.rs`: point it at the file that example
//! wrote and at the cached unquantized weights in turn, give both the same prompts, and the answers
//! sit side by side. Sampling is the framework's `Argmax`, so a run is deterministic and the only
//! thing that differs between the two is the weights.
//!
//! ```text
//! cargo run --release --example generate --features llama3,test-metal -- \
//!     --checkpoint /tmp/llama-3.2-1b-instruct-q4fb32.bpk \
//!     "What is the capital of France? One sentence."
//! ```
//!
//! With no `--checkpoint` it loads the pretrained unquantized weights from the local cache, which
//! is the baseline half of that comparison.

use std::time::Instant;

use burn::tensor::Device;
use burn_lm_inference::{Argmax, GeneratedItemEmitter, TextGenerationListener};
use burn_lm_llama::{pretrained::ModelMeta, LlamaConfig, LlamaVersion};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Greedily generate from a Llama-3.2-1B-Instruct checkpoint")]
struct Args {
    /// A local record to load (burnpack, or a legacy `.mpk`). Defaults to the cached
    /// unquantized pretrained weights.
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

fn main() {
    let args = Args::parse();
    let device = Device::default();
    println!("device: {device:?}");

    // The tokenizer always comes from the pretrained cache: it is the same file whatever the
    // weights were quantized to, and a local record carries weights only.
    let pretrained = LlamaVersion::Llama321bInstruct.pretrained();
    let tokenizer = pretrained
        .download_tokenizer()
        .expect("should have the tokenizer available");
    let checkpoint = match args.checkpoint {
        Some(path) => std::path::PathBuf::from(path),
        None => pretrained
            .download_weights()
            .expect("should have the unquantized weights available"),
    };
    println!("checkpoint: {}", checkpoint.display());

    let now = Instant::now();
    let mut llama = LlamaConfig::load_llama3_2_1b(
        checkpoint.to_str().unwrap(),
        tokenizer.to_str().unwrap(),
        args.max_seq_len,
        1,
        0,
        &device,
    )
    .expect("should load the model");
    println!("loaded in {:.1}s", now.elapsed().as_secs_f64());

    for prompt in &args.prompts {
        // The same single-turn chat template the Llama 3 server builds, so the model sees the
        // prompt framed the way it was instruction-tuned for.
        let framed = format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );

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
