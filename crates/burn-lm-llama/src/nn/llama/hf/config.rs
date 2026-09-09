//! Reading a repo's own `config.json` and turning it into a [`LlamaConfig`].
//!
//! This is the half of direct loading that removes the hand-written dimensions. Every Hugging Face
//! model repo carries its architecture's config as JSON — the same fields `vllm` and SGLang read —
//! so a fine-tune with a different vocabulary or a distill with fewer layers needs no code change,
//! only its own `config.json`.
//!
//! Two fields are worth their own paragraph.
//!
//! `rope_scaling` has accumulated several incompatible spellings over the years: the Llama-3.1
//! object keyed on `rope_type`, an older one keyed on `type` with only a `factor`, a bare float
//! from before either, and the `yarn`/`longrope` families that are a different computation
//! altogether. Only the `llama3` shape is one our `RopeFrequencyScaling` implements, so it is the
//! only one accepted; everything else is refused by name. Silently defaulting an unrecognised
//! scaling to "no scaling" would load and generate fluent nonsense at long context, which is the
//! worst possible failure.
//!
//! `head_dim` is optional in the file, and our attention derives it as `hidden_size /
//! num_attention_heads`. When a repo states one that disagrees with that division, the module we
//! would build is not the model the weights describe, so that too is refused rather than ignored.

use serde::Deserialize;

use crate::nn::pos_encoding::{RopeConfig, RopeFrequencyScaling};
use crate::LlamaConfig;

/// The architectures this loader knows how to build a module for.
///
/// One entry, because we have one architecture. The list exists so the refusal can say what it
/// does support, and so adding the second one is a line here rather than a new decision.
const SUPPORTED_ARCHITECTURES: [&str; 1] = ["LlamaForCausalLM"];

/// The subset of a repo's `config.json` that decides the shape of the module.
///
/// Unknown keys are ignored — a config carries plenty that inference does not care about
/// (`initializer_range`, `transformers_version`, a trainer's own bookkeeping) and rejecting a file
/// for having them would break on every new `transformers` release.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    /// The model classes this checkpoint was saved from. Absent in some hand-rolled configs, which
    /// is why `model_type` is also consulted.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// The architecture family, e.g. `llama`.
    #[serde(default)]
    pub model_type: Option<String>,
    /// Residual stream width — our `d_model`.
    pub hidden_size: usize,
    /// Feed-forward inner width — our `hidden_size`.
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Absent means "as many as there are attention heads", i.e. no grouped-query attention.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Left as raw JSON so the several historical spellings can be told apart and refused by name
    /// rather than failing to deserialize into one assumed shape.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    /// Stated head width. Optional; checked against `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Whether the output projection reuses the embedding matrix. True for Llama-3.2, and the
    /// reason those checkpoints carry no `lm_head.weight`.
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    /// The dtype the weights are stored in, for reporting. bf16 for every Llama-3 repo.
    #[serde(default)]
    pub torch_dtype: Option<String>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f32 {
    10000.0
}

impl HfConfig {
    /// Parse a repo's `config.json`.
    pub fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|err| format!("could not read config.json: {err}"))
    }

    /// Read a repo's `config.json` off disk.
    pub fn read(path: &std::path::Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|err| format!("could not read {}: {err}", path.display()))?;
        Self::parse(&json)
    }

    /// The number of key-value heads, resolved.
    pub fn kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Refuse a checkpoint this loader cannot honestly build.
    ///
    /// Reading safetensors gives us every Llama-shaped checkpoint, not every checkpoint: a Qwen or
    /// a Mistral has different norms and biases, and loading its weights into our Llama would place
    /// most of them successfully and be wrong. So the architecture is checked before anything is
    /// allocated.
    fn check_architecture(&self) -> Result<(), String> {
        if self
            .architectures
            .iter()
            .any(|arch| SUPPORTED_ARCHITECTURES.contains(&arch.as_str()))
        {
            return Ok(());
        }
        // Some configs carry no `architectures` at all. `model_type` is the same claim, weaker.
        if self.architectures.is_empty() && self.model_type.as_deref() == Some("llama") {
            return Ok(());
        }
        Err(format!(
            "unsupported architecture {:?} (model_type {:?}); this loader builds {:?}",
            self.architectures, self.model_type, SUPPORTED_ARCHITECTURES
        ))
    }

    /// The head width, checked against the one our attention derives.
    fn check_head_dim(&self) -> Result<(), String> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                self.hidden_size, self.num_attention_heads
            ));
        }
        let derived = self.hidden_size / self.num_attention_heads;
        match self.head_dim {
            Some(stated) if stated != derived => Err(format!(
                "config states head_dim {stated} but hidden_size / num_attention_heads is \
                 {derived}; this loader builds heads of the derived width only"
            )),
            _ => Ok(()),
        }
    }

    /// The RoPE frequency scaling this config asks for, or `None` for unscaled.
    ///
    /// Every shape that is not the Llama-3 one is an error naming what was found.
    pub fn rope_frequency_scaling(&self) -> Result<Option<RopeFrequencyScaling>, String> {
        let Some(scaling) = &self.rope_scaling else {
            return Ok(None);
        };
        if scaling.is_null() {
            return Ok(None);
        }
        let Some(object) = scaling.as_object() else {
            return Err(format!(
                "rope_scaling is {scaling}, not an object; the bare-factor spelling predates the \
                 llama3 scheme and is not supported"
            ));
        };

        // `rope_type` is the current key; `type` is what older configs wrote. Both can be present.
        let kind = object
            .get("rope_type")
            .or_else(|| object.get("type"))
            .and_then(|value| value.as_str());

        match kind {
            // Some configs spell "no scaling" as an object saying so.
            None | Some("default") => Ok(None),
            Some("llama3") => {
                let factor = number(object, "factor").ok_or_else(|| {
                    "rope_scaling is llama3 but carries no numeric factor".to_string()
                })?;
                let mut freq = RopeFrequencyScaling::new().with_scale_factor(factor);
                if let Some(low) = number(object, "low_freq_factor") {
                    freq = freq.with_low_freq_factor(low);
                }
                if let Some(high) = number(object, "high_freq_factor") {
                    freq = freq.with_high_freq_factor(high);
                }
                if let Some(old) = number(object, "original_max_position_embeddings") {
                    freq = freq.with_old_context_len(old);
                }
                Ok(Some(freq))
            }
            Some(other) => Err(format!(
                "rope_scaling type {other:?} is not implemented; only the llama3 scheme is \
                 (loading it as unscaled would generate fluent nonsense past the original context)"
            )),
        }
    }

    /// Build the [`LlamaConfig`] this checkpoint describes.
    ///
    /// `tokenizer_path` is the only thing that does not come out of the file: the module owns its
    /// tokenizer, and which file that is depends on how the repo ships it.
    pub fn to_llama_config(&self, tokenizer_path: &str) -> Result<LlamaConfig, String> {
        self.check_architecture()?;
        self.check_head_dim()?;
        let scaled = self.rope_frequency_scaling()?;

        Ok(LlamaConfig::new(
            self.intermediate_size,
            self.vocab_size,
            tokenizer_path.to_string(),
        )
        .with_d_model(self.hidden_size)
        .with_num_hidden_layers(self.num_hidden_layers)
        .with_num_attention_heads(self.num_attention_heads)
        .with_num_key_value_heads(Some(self.kv_heads()))
        .with_norm_eps(self.rms_norm_eps)
        .with_rope(RopeConfig::new(self.rope_theta).with_scaled(scaled)))
    }
}

/// A JSON number as `f32`, whichever way it was written. `8` and `8.0` are the same factor.
fn number(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<f32> {
    object.get(key)?.as_f64().map(|value| value as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `unsloth/Llama-3.2-1B-Instruct`'s own `config.json`, verbatim, so the comparison below runs
    /// with no network.
    const LLAMA_3_2_1B_CONFIG: &str = r#"{
      "architectures": ["LlamaForCausalLM"],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": 128000,
      "eos_token_id": 128009,
      "head_dim": 64,
      "hidden_act": "silu",
      "hidden_size": 2048,
      "initializer_range": 0.02,
      "intermediate_size": 8192,
      "max_position_embeddings": 131072,
      "mlp_bias": false,
      "model_type": "llama",
      "num_attention_heads": 32,
      "num_hidden_layers": 16,
      "num_key_value_heads": 8,
      "pad_token_id": 128004,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
      },
      "rope_theta": 500000.0,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.52.0.dev0",
      "unsloth_fixed": true,
      "use_cache": true,
      "vocab_size": 128256
    }"#;

    /// The claim the whole config-derivation step rests on: what the repo says about itself is what
    /// we had hand-written. If this ever disagrees, one of the two is wrong about the model.
    #[test]
    fn derived_config_matches_the_hardcoded_llama_3_2_1b() {
        let derived = HfConfig::parse(LLAMA_3_2_1B_CONFIG)
            .expect("fixture should parse")
            .to_llama_config("tokenizer.json")
            .expect("fixture should convert");
        let hardcoded = LlamaConfig::llama3_2_1b("tokenizer.json");

        assert_eq!(derived.d_model, hardcoded.d_model);
        assert_eq!(derived.hidden_size, hardcoded.hidden_size);
        assert_eq!(derived.num_hidden_layers, hardcoded.num_hidden_layers);
        assert_eq!(derived.num_attention_heads, hardcoded.num_attention_heads);
        assert_eq!(derived.num_key_value_heads, hardcoded.num_key_value_heads);
        assert_eq!(derived.vocab_size, hardcoded.vocab_size);
        assert_eq!(derived.norm_eps, hardcoded.norm_eps);
        assert_eq!(derived.rope.theta, hardcoded.rope.theta);

        let derived_scaling = derived.rope.scaled.expect("llama3 scaling");
        let hardcoded_scaling = hardcoded.rope.scaled.expect("llama3 scaling");
        assert_eq!(derived_scaling.scale_factor, hardcoded_scaling.scale_factor);
        assert_eq!(
            derived_scaling.low_freq_factor,
            hardcoded_scaling.low_freq_factor
        );
        assert_eq!(
            derived_scaling.high_freq_factor,
            hardcoded_scaling.high_freq_factor
        );
        assert_eq!(
            derived_scaling.old_context_len,
            hardcoded_scaling.old_context_len
        );
    }

    #[test]
    fn tied_embeddings_and_dtype_are_read() {
        let config = HfConfig::parse(LLAMA_3_2_1B_CONFIG).unwrap();
        assert!(config.tie_word_embeddings);
        assert_eq!(config.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(config.kv_heads(), 8);
    }

    #[test]
    fn absent_kv_heads_means_one_per_attention_head() {
        let json = r#"{"architectures":["LlamaForCausalLM"],"hidden_size":64,
            "intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
            "vocab_size":32}"#;
        let config = HfConfig::parse(json).unwrap();
        assert_eq!(config.kv_heads(), 4);
        // Unstated theta and epsilon fall back to the pre-Llama-3 defaults.
        let llama = config.to_llama_config("t").unwrap();
        assert_eq!(llama.rope.theta, 10000.0);
        assert!(llama.rope.scaled.is_none());
        assert_eq!(llama.norm_eps, 1e-5);
    }

    #[test]
    fn a_foreign_architecture_is_refused() {
        let json = r#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2",
            "hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
            "num_attention_heads":4,"vocab_size":32}"#;
        let err = HfConfig::parse(json)
            .unwrap()
            .to_llama_config("t")
            .unwrap_err();
        assert!(err.contains("Qwen2ForCausalLM"), "{err}");
    }

    /// The footgun this exists for: an unrecognised scaling must not quietly become "unscaled".
    #[test]
    fn unknown_rope_scaling_shapes_are_refused_by_name() {
        for (scaling, needle) in [
            (r#"{"type":"linear","factor":4.0}"#, "linear"),
            (r#"{"rope_type":"yarn","factor":4.0}"#, "yarn"),
            (r#"{"rope_type":"dynamic","factor":4.0}"#, "dynamic"),
            (r#"8.0"#, "not an object"),
        ] {
            let json = format!(
                r#"{{"architectures":["LlamaForCausalLM"],"hidden_size":64,
                "intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
                "vocab_size":32,"rope_scaling":{scaling}}}"#
            );
            let err = HfConfig::parse(&json)
                .unwrap()
                .to_llama_config("t")
                .unwrap_err();
            assert!(err.contains(needle), "{scaling} gave {err}");
        }
    }

    #[test]
    fn a_default_rope_scaling_object_means_unscaled() {
        let json = r#"{"architectures":["LlamaForCausalLM"],"hidden_size":64,
            "intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
            "vocab_size":32,"rope_scaling":{"rope_type":"default"}}"#;
        let llama = HfConfig::parse(json).unwrap().to_llama_config("t").unwrap();
        assert!(llama.rope.scaled.is_none());
    }

    #[test]
    fn a_stated_head_dim_that_disagrees_is_refused() {
        let json = r#"{"architectures":["LlamaForCausalLM"],"hidden_size":64,
            "intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,
            "vocab_size":32,"head_dim":128}"#;
        let err = HfConfig::parse(json)
            .unwrap()
            .to_llama_config("t")
            .unwrap_err();
        assert!(err.contains("head_dim 128"), "{err}");
    }
}
