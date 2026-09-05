use std::path::{Path, PathBuf};

use burn::prelude::*;

use super::{inference::Llama, LlamaConfig, LlamaVersion, TinyLlamaVersion};
use crate::nn::transformer::Transformer;

#[cfg(feature = "llama3")]
use crate::tokenizer::Tiktoken;

#[cfg(feature = "tiny")]
use crate::tokenizer::SentencePieceTokenizer;

/// Pre-trained model metadata.
pub struct Pretrained {
    pub(super) name: &'static str,
    pub(super) model: &'static str,
    pub(super) tokenizer: &'static str,
}

mod downloader {
    use super::*;
    use burn::data::network::downloader;
    use std::fs::{create_dir_all, File};
    use std::io::Write;

    impl Pretrained {
        fn model_dir(&self) -> PathBuf {
            dirs::home_dir()
                .expect("Should be able to get home directory")
                .join(".cache")
                .join("llama")
                .join(self.name)
        }

        fn model_file_name(&self, url: &str) -> String {
            url.rsplit_once('/')
                .unwrap()
                .1
                .replace("?download=true", "")
        }

        pub fn is_downloaded(&self) -> bool {
            let model_name = self.model_dir().join(self.model_file_name(self.model));
            let tokenizer_name = self.model_dir().join(self.model_file_name(self.tokenizer));
            model_name.exists() && tokenizer_name.exists()
        }

        /// Download the file to the local cache directory.
        fn download(&self, url: &str) -> Result<PathBuf, std::io::Error> {
            // Model cache directory
            let model_dir = self.model_dir();

            if !model_dir.exists() {
                create_dir_all(&model_dir)?;
            }

            let file_base_name = self.model_file_name(url);
            let file_name = model_dir.join(&file_base_name);
            if !file_name.exists() {
                // Download file content
                let bytes = downloader::download_file_as_bytes(url, &file_base_name);

                // Write content to file
                let mut output_file = File::create(&file_name)?;
                output_file.write_all(&bytes)?; // write_all is not OS limited (files over 2GB)
            }

            Ok(file_name)
        }

        /// Delete the file to the local cache directory.
        fn delete(&self, url: &str) -> Result<(), std::io::Error> {
            let model_dir = self.model_dir();
            if !model_dir.exists() {
                return Ok(());
            }
            let file_base_name = self.model_file_name(url);
            let file_name = model_dir.join(&file_base_name);
            if file_name.exists() {
                std::fs::remove_file(file_name)
                    .unwrap_or_else(|_| panic!("should delete model file '{file_base_name}'"));
            }
            Ok(())
        }

        /// Download the pre-trained model weights to the local cache directory.
        pub fn download_weights(&self) -> Result<PathBuf, std::io::Error> {
            self.download(self.model)
        }

        /// Delete the tokenizer to the local cache directory.
        pub fn download_tokenizer(&self) -> Result<PathBuf, std::io::Error> {
            self.download(self.tokenizer)
        }

        /// The file to load the weights from: the burnpack copy of the downloaded checkpoint once
        /// there is one, and the download itself otherwise.
        ///
        /// Every published artifact is still a legacy `.mpk` (see `legacy_mpk.rs`), and reading one
        /// means walking a whole msgpack document and converting every tensor on the way in —
        /// seconds of it, on every load. burnpack loads a fraction as slowly, so the first load
        /// after a download writes the weights back out beside the `.mpk` (`cache_burnpack`) and
        /// every load after that finds them here.
        pub fn checkpoint(&self) -> Result<PathBuf, std::io::Error> {
            let downloaded = self.download_weights()?;
            let burnpack = burnpack_sibling(&downloaded);
            if burnpack.exists() {
                return Ok(burnpack);
            }
            Ok(downloaded)
        }

        /// Delete the pre-trained model weights from the local cache directory.
        pub fn delete_weights(&self) -> Result<(), std::io::Error> {
            self.delete(self.model)
        }

        /// Delete the tokenizer from the local cache directory.
        pub fn delete_tokenizer(&self) -> Result<(), std::io::Error> {
            self.delete(self.tokenizer)
        }
    }
}

/// Where the burnpack copy of a cached checkpoint lives: the same path, with a `.bpk` extension.
fn burnpack_sibling(checkpoint: &Path) -> PathBuf {
    checkpoint.with_extension("bpk")
}

/// Write the freshly loaded weights back out as burnpack, beside the legacy checkpoint they came
/// from, so the next load reads that instead.
///
/// Called once per cached model — after a load from a `.mpk`, and never after a load that already
/// found the `.bpk`. It is best effort by design: the weights are in memory and the caller's load
/// has succeeded, so a cache that cannot be written (read-only directory, full disk) costs a slow
/// load next time and nothing else, which is what the current behavior already is.
///
/// The file appears atomically, written under a temporary name in the same directory and renamed
/// once complete, because a half-written `.bpk` is worse than no `.bpk` at all: `checkpoint` would
/// prefer it on the next run and the load would fail on a truncated file. The rename is within one
/// directory, so it is the filesystem's own atomic swap.
fn cache_burnpack(model: &Transformer, loaded_from: &Path) {
    // Nothing to do when the load already took the fast path.
    if loaded_from.extension().is_none_or(|ext| ext != "mpk") {
        return;
    }

    let target = burnpack_sibling(loaded_from);
    // The pid keeps two processes converting the same cache from writing to one another's file.
    let temp = target.with_extension(format!("bpk.{}.tmp", std::process::id()));

    let written = model
        .clone()
        .save_file(&temp)
        .map_err(|err| err.to_string())
        .and_then(|()| std::fs::rename(&temp, &target).map_err(|err| err.to_string()));

    // Printed rather than logged, like the record save and load this sits between (see
    // `Llama::save`): only the HTTP server installs a tracing subscriber, and a load that quietly
    // spends seconds writing several gigabytes next to the checkpoint should say so.
    match written {
        Ok(()) => println!(
            "Converted {} to {}. Later loads read the burnpack copy.",
            loaded_from.display(),
            target.display()
        ),
        Err(err) => {
            let _ = std::fs::remove_file(&temp);
            eprintln!(
                "Could not write {}: {err}. The model is loaded; later loads keep reading {}.",
                target.display(),
                loaded_from.display()
            );
        }
    }
}

pub trait ModelMeta {
    fn pretrained(&self) -> Pretrained;
}

/// The local Q4 record to load in place of the published artifact, from
/// `BURN_LM_LLAMA_Q4_CHECKPOINT`.
///
/// The published `q4fb32` `.mpk` stores its quantized tensors in a layout burn 0.22's reader
/// refuses, so the working Q4 checkpoint is one produced locally (`cargo run --example quantize`)
/// and named through this variable. Both the loader and the "is it downloaded?" check read it here,
/// so a deployment pointing at a local record doesn't get told to download anything.
pub fn q4_checkpoint_override() -> Option<std::path::PathBuf> {
    std::env::var("BURN_LM_LLAMA_Q4_CHECKPOINT")
        .ok()
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from)
}

impl ModelMeta for LlamaVersion {
    fn pretrained(&self) -> Pretrained {
        match self {
            Self::Llama3Instruct => Pretrained {
                name: "Llama-3-8B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
            Self::Llama31Instruct => Pretrained {
                name: "Llama-3.1-8B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
            Self::Llama323bInstruct => Pretrained {
                name: "Llama-3.2-3B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
            Self::Llama321bInstruct => Pretrained {
                name: "Llama-3.2-1B-Instruct",
                model: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/tokenizer.model?download=true",
            },
            Self::Llama321bInstructQ4FB32 => Pretrained {
                name: "Llama-3.2-1B-Instruct-Q4",
                model: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-q4fb32-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-q4fb32-burn/resolve/main/tokenizer.model?download=true",
            },
        }
    }
}

impl ModelMeta for TinyLlamaVersion {
    fn pretrained(&self) -> Pretrained {
        match self {
            TinyLlamaVersion::V1 => Pretrained {
                name: "TinyLlama-1.1B",
                model: "https://huggingface.co/tracel-ai/tiny-llama-1.1b-burn/resolve/main/model.mpk?download=true",
                tokenizer: "https://huggingface.co/tracel-ai/tiny-llama-1.1b-burn/resolve/main/tokenizer.json?download=true",
            }
        }
    }
}

fn check_context_length(max_seq_len: usize, max_context_len: usize) {
    assert!(
        max_seq_len <= max_context_len,
        "Maximum sequence length must not exceed {max_context_len}"
    );
}

impl LlamaConfig {
    /// Load pre-trained Llama-3.2-3B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `max_slots` - The number of concurrent sequences (KV slab lanes) to size for.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_2_3b_pretrained(
        max_seq_len: usize,
        max_slots: usize,
        kv_pool_tokens: usize,
        device: &Device,
    ) -> Result<Llama<Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = LlamaVersion::Llama323bInstruct.pretrained();
        let checkpoint = model
            .checkpoint()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        // `max_slots` sizes the shared KV slab: one lane per concurrent sequence the batched server
        // can admit. The 3b server passes its per-model slot count here (see its `Default`).
        let llama = Self::load_llama3_2_3b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            max_slots,
            kv_pool_tokens,
            device,
        )?;
        // Leave a burnpack copy behind when this came from the legacy `.mpk`.
        cache_burnpack(&llama.decoder.model, &checkpoint);
        Ok(llama)
    }

    /// Load pre-trained Llama-3.2-3B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_2_1b_pretrained(
        max_seq_len: usize,
        max_batch_size: usize,
        kv_pool_tokens: usize,
        device: &Device,
    ) -> Result<Llama<Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = LlamaVersion::Llama321bInstruct.pretrained();
        let checkpoint = model
            .checkpoint()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        let llama = Self::load_llama3_2_1b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            max_batch_size,
            kv_pool_tokens,
            device,
        )?;
        // Leave a burnpack copy behind when this came from the legacy `.mpk`.
        cache_burnpack(&llama.decoder.model, &checkpoint);
        Ok(llama)
    }

    /// Load the 4-bit quantized Llama-3.2-1B-Instruct model with
    /// [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// Set `BURN_LM_LLAMA_Q4_CHECKPOINT` to a local record to load that instead of the published
    /// artifact — which is what you want today, since the published one is in a quantization format
    /// the current reader no longer accepts. `cargo run --example quantize` writes such a record.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_2_1b_pretrained_q4(
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Llama<Tiktoken>, String> {
        // Llama-3.2 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // The published Q4 artifact is an old-format `.mpk` whose quantized tensors the current
        // reader refuses (burn 0.22 reworked `QuantScheme`), so a working Q4 checkpoint has to be
        // produced locally — `cargo run --example quantize` does it from the unquantized weights.
        // `BURN_LM_LLAMA_Q4_CHECKPOINT` is how that local file gets used: point it at the record and
        // this loads it instead of reaching for the published one. The tokenizer still comes from
        // the unquantized repo, since it is the same file whatever the weights were quantized to and
        // a local record carries weights only.
        let (checkpoint, tokenizer) = match q4_checkpoint_override() {
            Some(path) => {
                let tokenizer = LlamaVersion::Llama321bInstruct
                    .pretrained()
                    .download_tokenizer()
                    .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;
                (path, tokenizer)
            }
            None => {
                let model = LlamaVersion::Llama321bInstructQ4FB32.pretrained();
                let checkpoint = model
                    .checkpoint()
                    .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
                let tokenizer = model
                    .download_tokenizer()
                    .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;
                (checkpoint, tokenizer)
            }
        };

        // The Q4 server is single-shot (not a `BatchedInferenceServer`), so it only ever drives lane
        // 0. A single-lane slab is correct and avoids eagerly allocating KV for lanes it can never
        // use — the opposite of what a quantized, memory-saving model wants.
        let llama = Self::load_llama3_2_1b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            1,
            0, // window-per-lane pool: one lane, one window — nothing to oversubscribe
            device,
        )?;
        // Leave a burnpack copy behind when this came from the legacy `.mpk`.
        cache_burnpack(&llama.decoder.model, &checkpoint);
        Ok(llama)
    }

    /// Load pre-trained Llama-3.1-8B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `max_slots` - The number of concurrent sequences (KV slab lanes) to size for.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_1_8b_pretrained(
        max_seq_len: usize,
        max_slots: usize,
        kv_pool_tokens: usize,
        device: &Device,
    ) -> Result<Llama<Tiktoken>, String> {
        // Llama-3.1 models support context length up to 128K tokens.
        check_context_length(max_seq_len, 128 * 1024);

        // Download checkpoint and tokenizer
        let model = LlamaVersion::Llama31Instruct.pretrained();
        let checkpoint = model
            .checkpoint()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        // `max_slots` sizes the shared KV slab: one lane per concurrent sequence the batched server
        // can admit. The 3.1-8b server passes its per-model slot count here (see its `Default`).
        let llama = Self::load_llama3_1_8b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            max_slots,
            kv_pool_tokens,
            device,
        )?;
        // Leave a burnpack copy behind when this came from the legacy `.mpk`.
        cache_burnpack(&llama.decoder.model, &checkpoint);
        Ok(llama)
    }

    /// Load pre-trained Llama-3-8B-Instruct model with [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    ///
    /// # Arguments
    /// - `max_seq_len` - The maximum sequence length for input text.
    /// - `max_slots` - The number of concurrent sequences (KV slab lanes) to size for.
    /// - `device` - The device to load the model on.
    #[cfg(feature = "llama3")]
    pub fn llama3_8b_pretrained(
        max_seq_len: usize,
        max_slots: usize,
        kv_pool_tokens: usize,
        device: &Device,
    ) -> Result<Llama<Tiktoken>, String> {
        // Llama-3 models support context length up to 8K tokens.
        check_context_length(max_seq_len, 8 * 1024);

        // Download checkpoint and tokenizer
        let model = LlamaVersion::Llama3Instruct.pretrained();
        let checkpoint = model
            .checkpoint()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        // `max_slots` sizes the shared KV slab: one lane per concurrent sequence the batched server
        // can admit. The 8b server passes its per-model slot count here (see its `Default`).
        let llama = Self::load_llama3_8b(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            max_slots,
            kv_pool_tokens,
            device,
        )?;
        // Leave a burnpack copy behind when this came from the legacy `.mpk`.
        cache_burnpack(&llama.decoder.model, &checkpoint);
        Ok(llama)
    }

    /// Load pre-trained TinyLlama-1.1B Chat v1.0 model with [SentenciePiece](https://github.com/google/sentencepiece) tokenizer.
    #[cfg(feature = "tiny")]
    pub fn tiny_llama_pretrained(
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Llama<SentencePieceTokenizer>, String> {
        // TinyLlama models support context length up to 2K tokens.

        check_context_length(max_seq_len, 2 * 1024);

        // Download checkpoint and tokenizer
        let model = TinyLlamaVersion::V1.pretrained();
        let checkpoint = model
            .checkpoint()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
        let tokenizer = model
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        let llama = Self::load_tiny_llama(
            checkpoint.to_str().unwrap(),
            tokenizer.to_str().unwrap(),
            max_seq_len,
            device,
        )?;
        // Leave a burnpack copy behind when this came from the legacy `.mpk`.
        cache_burnpack(&llama.decoder.model, &checkpoint);
        Ok(llama)
    }
}
