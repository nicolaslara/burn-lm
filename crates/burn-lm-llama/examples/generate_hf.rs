//! Generate greedily from a Llama read straight out of a Hugging Face repo.
//!
//! This is the direct-loading counterpart to `examples/generate.rs`: same greedy sampling, same
//! chat framing, same reporting, but the weights come from the repo's own `model.safetensors` and
//! the dimensions from its own `config.json` rather than from a checkpoint someone converted into
//! burn's format and published for us.
//!
//! ```text
//! cargo run --release --example generate-hf --features hf,test-metal -- \
//!     --repo unsloth/Llama-3.2-1B-Instruct \
//!     "What is the capital of France? One sentence."
//! ```
//!
//! Because sampling is `Argmax`, running the same prompts through `generate.rs` on the published
//! `.mpk` gives the comparison that says whether this loaded the weights correctly: the two
//! transcripts should agree, and fluent-but-different text means the layout is wrong, not that the
//! model is creative.
//!
//! `--config-only` derives and prints the config without fetching any weights, which is the cheap
//! way to see what a repo would build.

use std::time::Instant;

use burn::tensor::Device;
use burn_lm_inference::{Argmax, GeneratedItemEmitter, TextGenerationListener};
use burn_lm_llama::hf::{load_hf, HfConfig, HfRepo};
use burn_lm_llama::inference::Llama;
use burn_lm_llama::tokenizer::Tokenizer;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "Greedily generate from a Hugging Face Llama repo")]
struct Args {
    /// The repo id, `owner/name`.
    #[arg(short, long, default_value = "unsloth/Llama-3.2-1B-Instruct")]
    repo: String,

    /// A branch, tag, or commit sha. Defaults to the repo's default branch.
    #[arg(long)]
    revision: Option<String>,

    /// Derive and print the config from the repo's `config.json` and stop, without downloading
    /// weights.
    #[arg(long, default_value_t = false)]
    config_only: bool,

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
    println!("repo: {}", args.repo);

    if args.config_only {
        report_config(&args);
        return;
    }

    let now = Instant::now();
    let llama = load_hf(
        &args.repo,
        args.revision.as_deref(),
        args.max_seq_len,
        1,
        0,
        &device,
    )
    .expect("should load the model");
    run(llama, now, &args);
}

/// Fetch only the config and print what it would build, next to the fields it was derived from.
fn report_config(args: &Args) {
    let mut repo = HfRepo::new(&args.repo);
    if let Some(revision) = &args.revision {
        repo = repo.with_revision(revision.clone());
    }
    let path = repo.file("config.json").expect("should fetch config.json");
    let hf = HfConfig::read(&path).expect("should read config.json");
    let derived = hf
        .to_llama_config("<tokenizer.json>")
        .expect("should derive a config");

    println!("torch_dtype: {:?}", hf.torch_dtype);
    println!("tie_word_embeddings: {}", hf.tie_word_embeddings);
    println!("d_model: {}", derived.d_model);
    println!("hidden_size: {}", derived.hidden_size);
    println!("num_hidden_layers: {}", derived.num_hidden_layers);
    println!("num_attention_heads: {}", derived.num_attention_heads);
    println!("num_key_value_heads: {:?}", derived.num_key_value_heads);
    println!("vocab_size: {}", derived.vocab_size);
    println!("norm_eps: {}", derived.norm_eps);
    println!("rope.theta: {}", derived.rope.theta);
    match &derived.rope.scaled {
        Some(scaling) => println!(
            "rope.scaled: factor {} low {} high {} old_context {}",
            scaling.scale_factor,
            scaling.low_freq_factor,
            scaling.high_freq_factor,
            scaling.old_context_len
        ),
        None => println!("rope.scaled: none"),
    }
}

/// Run every prompt through the model, reporting the load time measured from `loading`.
///
/// Deliberately the same shape as `generate.rs`'s `run`, so the two transcripts line up when read
/// side by side.
fn run<T: Tokenizer + 'static>(mut llama: Llama<T>, loading: Instant, args: &Args) {
    println!("loaded in {:.1}s", loading.elapsed().as_secs_f64());

    for prompt in &args.prompts {
        let framed = frame(prompt);

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

/// The Llama-3 single-turn chat framing, the same string `generate.rs` and `server/llama3.rs`
/// build. A repo's `tokenizer_config.json` carries a Jinja `chat_template` that would render this
/// from the repo itself; using it is a separate decision (see the module docs on `hf`).
fn frame(prompt: &str) -> String {
    format!(
        "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>\
         <|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}
