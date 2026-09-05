//! Re-quantize Llama-3.2-1B-Instruct from the f16/f32 checkpoint we can still read, and save the
//! result in burn's current record format.
//!
//! Why this exists: burn 0.22 reworked `QuantScheme` into a two-level scale layout, so the
//! published `q4fb32` `.mpk` artifact no longer describes itself in terms the current reader
//! understands — and our legacy `.mpk` reader deliberately refuses `QFloat` data rather than
//! guessing at the new fields. The unquantized checkpoint is plain float data, which the legacy
//! reader still handles, so the way back to a working Q4 model is to load that and quantize here.
//!
//! Defaults reproduce the semantics the old artifact's name spelled out — `q4fb32` is `Q4F` values
//! with one scale per block of 32 — and the knobs exist because which scale dtype and store a
//! given backend can actually execute is a property of the backend, not of the model.
//!
//! ```text
//! cargo run --release --example quantize \
//!     --features llama3,test-metal -- --output /tmp/llama-3.2-1b-instruct-q4fb32.bpk
//! ```
//!
//! Quantizing a 4-bit scheme needs a backend whose `quantize` implements it: the cubecl backends
//! (metal, cuda, wgpu, rocm) do, while burn's ndarray backend only implements 8-bit, so a Q4 run
//! on the default CPU device stops at an `unimplemented!` inside burn. The device comes from
//! `Device::default()`, which honours `BURN_DEVICE`, so the backend is chosen by the feature set
//! this example is built with rather than by anything hardcoded here.

use std::path::Path;
use std::time::Instant;

use burn::tensor::{
    quantization::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
    Device,
};
use burn_lm_llama::{pretrained::ModelMeta, LlamaConfig, LlamaVersion};
use clap::{Parser, ValueEnum};

/// The KV window to build the model with. Quantization never runs a forward pass, so the cache is
/// dead weight here; keep it small so the process pays for weights only.
const MAX_SEQ_LEN: usize = 512;

#[derive(Parser, Debug)]
#[command(about = "Quantize the Llama-3.2-1B-Instruct weights and save them as a burnpack record")]
struct Args {
    /// Where to write the quantized record. A path with no extension gets `.bpk`.
    #[arg(short, long)]
    output: String,

    /// The quantized value type. `q4f` is the 4-bit full-range type the published artifact used.
    #[arg(long, value_enum, default_value_t = Value::Q4f)]
    value: Value,

    /// How many values share one scale. `0` means a single scale for the whole tensor.
    #[arg(long, default_value_t = 32)]
    block_size: u8,

    /// The dtype the scales are stored in.
    #[arg(long, value_enum, default_value_t = Scale::F32)]
    scale_dtype: Scale,

    /// Store the packed values natively instead of packing them into `u32` words. Only some
    /// value types have a native sub-byte representation; 4-bit values need the `u32` packing.
    #[arg(long)]
    native_store: bool,
}

/// The quantized value types worth exposing here: full-range and symmetric variants at 4 and 8
/// bits. The rest of `QuantValue` (the float and 2-bit types) is not what a Llama checkpoint is
/// asking for, so it stays out of the surface.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Value {
    Q4f,
    Q4s,
    Q8f,
    Q8s,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Scale {
    F32,
    F16,
    Bf16,
}

impl From<Value> for QuantValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Q4f => QuantValue::Q4F,
            Value::Q4s => QuantValue::Q4S,
            Value::Q8f => QuantValue::Q8F,
            Value::Q8s => QuantValue::Q8S,
        }
    }
}

impl From<Scale> for ScaleDtype {
    fn from(scale: Scale) -> Self {
        match scale {
            Scale::F32 => ScaleDtype::F32,
            Scale::F16 => ScaleDtype::F16,
            Scale::Bf16 => ScaleDtype::BF16,
        }
    }
}

fn main() {
    let args = Args::parse();

    // One scale level, chosen by `--block-size`. A two-level scheme (narrow block scales
    // normalized by a per-tensor scale) only pays off when the block dtype is narrower than the
    // tensor one, which none of the options here are, so there is nothing to gain from adding the
    // second level and it stays off.
    let scheme = QuantScheme::default()
        .with_value(args.value.into())
        .with_store(if args.native_store {
            QuantStore::Native
        } else {
            // Packing dim 0 counts from the innermost dimension, so 4-bit values pack along the
            // rows they are read across.
            QuantStore::PackedU32(0)
        });
    let scheme = if args.block_size == 0 {
        scheme.per_tensor(args.scale_dtype.into())
    } else {
        scheme.per_block([args.block_size], args.scale_dtype.into())
    };

    let device = Device::default();
    println!("device: {device:?}");
    println!("scheme: {scheme:?}");

    // The unquantized weights and the tokenizer, from the local cache (downloaded on first use).
    let pretrained = LlamaVersion::Llama321bInstruct.pretrained();
    let checkpoint = pretrained
        .download_weights()
        .expect("should have the unquantized weights available");
    let input_bytes = file_size(&checkpoint);
    println!(
        "input:  {} ({})",
        checkpoint.display(),
        human_size(input_bytes)
    );

    let now = Instant::now();
    // One lane, no oversubscribed pool: this process only ever loads and rewrites weights.
    let llama = LlamaConfig::llama3_2_1b_pretrained(MAX_SEQ_LEN, 1, 0, &device)
        .expect("should load the unquantized model");
    println!("loaded in {:.1}s", now.elapsed().as_secs_f64());

    let now = Instant::now();
    let llama = llama.quantize(scheme);
    let _ = device.sync();
    println!("quantized in {:.1}s", now.elapsed().as_secs_f64());

    llama.save(&args.output).expect("should save the record");

    // `save` appends `.bpk` when the path carries no extension, so ask the filesystem which file
    // actually appeared rather than assuming the argument names it.
    let output = match Path::new(&args.output).extension() {
        Some(_) => args.output.clone(),
        None => format!("{}.bpk", args.output),
    };
    let output_bytes = file_size(Path::new(&output));
    println!("output: {output} ({})", human_size(output_bytes));
    println!(
        "compression: {:.2}x",
        input_bytes as f64 / output_bytes as f64
    );
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path)
        .map(|meta| meta.len())
        .unwrap_or_else(|err| panic!("should stat {}: {err}", path.display()))
}

fn human_size(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}
