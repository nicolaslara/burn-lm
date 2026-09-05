//! Reader for PyTorch `state_dict` checkpoints (the `import` feature).
//!
//! This used to go through `burn-import`'s `PyTorchFileRecorder`, which deserialized the file into
//! a generated `TransformerRecord` and handed that to `load_record`. burn dropped both the
//! `*Record` types and that recorder, so the same job is now done by `burn_store::PytorchStore`:
//! it reads the file, rewrites the checkpoint's keys into our module's parameter paths, and
//! applies the tensors straight onto an already-`init`ed module — the same applier the legacy
//! `.mpk` reader uses (see `legacy_mpk`). The key remappings below are unchanged; only the
//! builder's name changed (`with_key_remap` -> `with_key_remapping`), the regex and replacement
//! syntax is the same. Transposing PyTorch's linear weights and matching PyTorch's `norm.weight`
//! against our `norm.gamma` is still handled for us, by the store's `PyTorchToBurnAdapter`.

use burn::prelude::*;
use burn_store::{ModuleSnapshot, PytorchStore};

use crate::tokenizer::Tokenizer;

use super::{inference::Llama, LlamaConfig};

impl LlamaConfig {
    /// Load pre-trained Llama checkpoint.
    pub fn load_pretrained<T: Tokenizer>(
        &self,
        checkpoint: &str,
        device: &Device,
    ) -> Result<Llama<T>, String> {
        let mut llama = self.init(device)?;

        // Load weights from torch state_dict
        let mut store = PytorchStore::from_file(checkpoint);

        if !cfg!(feature = "tiny") {
            store = store
                // Map layers.[i].feed_forward.w1.* -> layers.[i].feed_forward.swiglu.linear_inner.*
                .with_key_remapping(
                    "(layers\\.[0-9]+\\.feed_forward)\\.w1\\.(.+)",
                    "$1.swiglu.linear_inner.$2",
                )
                // Map layers.[i].feed_forward.w3.* -> layers.[i].feed_forward.swiglu.linear_outer.*
                .with_key_remapping(
                    "(layers\\.[0-9]+\\.feed_forward)\\.w3\\.(.+)",
                    "$1.swiglu.linear_outer.$2",
                )
                // Map norm.weight -> norm.gamma for all layers
                .with_key_remapping("(.*)norm\\.weight", "${1}norm.gamma");
        } else {
            store = store
                // Map lm_head.* -> output.*
                .with_key_remapping("lm_head\\.(.+)", "output.$1")
                // Remove model. prefix
                .with_key_remapping("model\\.(.+)", "$1")
                // Map embed_tokens.* -> tok_embeddings.*
                .with_key_remapping("embed_tokens\\.(.+)", "tok_embeddings.$1")
                // Map layers.[i].input_layernorm.* -> layers.[i].attention_norm.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.input_layernorm\\.(.+)",
                    "$1.attention_norm.$2",
                )
                // Map layers.[i].post_attention_layernorm.* -> layers.[i].ffn_norm.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.post_attention_layernorm\\.(.+)",
                    "$1.ffn_norm.$2",
                )
                // Map layers.[i].mlp.down_proj.* -> layers.[i].feed_forward.w2.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.mlp\\.down_proj\\.(.+)",
                    "$1.feed_forward.w2.$2",
                )
                // Map layers.[i].mlp.gate_proj.* -> layers.[i].feed_forward.swiglu.linear_inner.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.mlp\\.gate_proj\\.(.+)",
                    "$1.feed_forward.swiglu.linear_inner.$2",
                )
                // Map layers.[i].mlp.up_proj.* -> layers.[i].feed_forward.swiglu.linear_outer.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.mlp\\.up_proj\\.(.+)",
                    "$1.feed_forward.swiglu.linear_outer.$2",
                )
                // Map layers.[i].self_attn.k_proj.* -> layers.[i].attention.wk.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.self_attn\\.k_proj\\.(.+)",
                    "$1.attention.wk.$2",
                )
                // Map layers.[i].self_attn.o_proj.* -> layers.[i].attention.wo.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.self_attn\\.o_proj\\.(.+)",
                    "$1.attention.wo.$2",
                )
                // Map layers.[i].self_attn.q_proj.* -> layers.[i].attention.wq.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.self_attn\\.q_proj\\.(.+)",
                    "$1.attention.wq.$2",
                )
                // Map layers.[i].self_attn.v_proj.* -> layers.[i].attention.wv.*
                .with_key_remapping(
                    "(layers\\.[0-9]+)\\.self_attn\\.v_proj\\.(.+)",
                    "$1.attention.wv.$2",
                )
                // Map norm.weight -> norm.gamma for all layers
                .with_key_remapping("(.*)norm\\.weight", "${1}norm.gamma");
        }

        // The remapped keys are relative to the transformer, so the store is applied to it rather
        // than to the whole `Llama`. A checkpoint may legitimately carry tensors we have no
        // parameter for (the precomputed rotary frequencies, for one), so only the reverse —
        // a parameter the file does not cover — is an error, which is what the store's default
        // `allow_partial(false)` and `validate(true)` already enforce.
        llama
            .decoder
            .model
            .load_from(&mut store)
            .map_err(|err| format!("failed to apply {checkpoint}: {err}"))?;

        if cfg!(feature = "tiny") {
            // TinyLlama weights from HuggingFace use a different rotary positional encoding
            // which requires weight permutation:
            // https://github.com/huggingface/transformers/issues/25199#issuecomment-1687720247
            // https://github.com/jzhang38/TinyLlama/issues/24
            //
            // Without a record type to fix up before loading, the permutation now happens on the
            // module's parameters after the weights are in place. The tensors are the same
            // (already transposed to burn's `[d_in, d_out]` layout by the store's adapter, as they
            // were in the record), so the reshape/swap_dims sequence is unchanged.
            let n_heads = self.num_attention_heads;
            let n_kv_heads = self.num_key_value_heads.unwrap_or(n_heads);
            let wk_dim = self.d_model * n_kv_heads / n_heads;
            let permute = |w: Tensor<2>, n_heads: usize, dim1: usize, dim2: usize| {
                w // [2048, 256]
                    .reshape([dim1, n_heads, 2, dim2 / n_heads / 2]) // [2048, 4, 2, 32]
                    .swap_dims(2, 3) // [2048, 4, 32, 2]
                    .reshape([dim1, dim2])
            };

            let layers = core::mem::take(&mut llama.decoder.model.layers);
            llama.decoder.model.layers = layers
                .into_iter()
                .map(|mut layer| {
                    layer.attention.wq.weight = layer
                        .attention
                        .wq
                        .weight
                        .map(|w| permute(w, n_heads, self.d_model, self.d_model));
                    layer.attention.wk.weight = layer
                        .attention
                        .wk
                        .weight
                        .map(|w| permute(w, n_kv_heads, self.d_model, wk_dim));
                    layer
                })
                .collect::<Vec<_>>();
        }

        println!("Llama weights loaded");

        Ok(llama)
    }
}
