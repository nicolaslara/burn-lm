//! Applying a canonical Hugging Face safetensors file to our transformer.
//!
//! Three things stand between the file on the Hub and a working model, and each of them fails
//! quietly rather than loudly if you skip it — the model loads, generates fluent text, and is
//! wrong. They are worth naming.
//!
//! **Names.** A checkpoint calls the query projection `model.layers.0.self_attn.q_proj.weight`;
//! our module calls it `layers.0.attention.wq.weight`. The mapping is the one `import.rs` already
//! wrote for the PyTorch path — the same names, since a `state_dict` and a safetensors export of
//! the same model carry the same keys — so the rules here are that rule set pointed at a different
//! store.
//!
//! **Layout.** `SafetensorsStore` defaults to `IdentityAdapter`, unlike `PytorchStore`, which
//! installs `PyTorchToBurnAdapter` for you. A checkpoint exported from PyTorch stores a linear
//! weight as `[out, in]` and burn wants `[in, out]`, so without that adapter every linear layer in
//! the model loads transposed. And dtype conversion is opt-in too: these files are bf16 and the
//! applier refuses a dtype the module does not have, so a float cast to the device's own float type
//! is chained after it.
//!
//! **Rotation convention.** burn's `RotaryEncoding` rotates adjacent pairs of a head's channels —
//! Meta's original convention. Hugging Face's Llama rotates the two halves of a head against each
//! other, and its conversion script permutes `q_proj` and `k_proj` to match.
//! Reading HF weights therefore means undoing that permutation, which is what `import.rs` already
//! does for the HF-exported TinyLlama checkpoint. Get this wrong and the model is fluent and
//! subtly, worseningly wrong — the failure the design study warns about.
//!
//! There is a fourth, which is not a footgun so much as a structural fact: Llama-3.2 ties the
//! output projection to the embedding matrix, so those repos ship no `lm_head.weight` at all. Our
//! `Transformer` has a real `output` linear, so the tie is materialized after the load by
//! transposing the embedding into it.

use std::path::Path;

use burn::prelude::*;
use burn::tensor::DType;
use burn_store::SafetensorsStore;
use burn_store::{FloatCastAdapter, ModuleAdapter, ModuleSnapshot, PyTorchToBurnAdapter};

use crate::nn::transformer::Transformer;

/// The parameter a tied-embedding checkpoint does not carry.
const TIED_OUTPUT: &str = "output.weight";

/// How the model's output projection got its weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputWeights {
    /// The checkpoint carried its own `lm_head.weight`.
    FromCheckpoint,
    /// The checkpoint ties the output to the embedding, and the tie was materialized on load.
    TiedToEmbedding,
}

/// Read `path` and apply it to `model`.
///
/// `tied` is the config's `tie_word_embeddings`: it says whether a missing `output.weight` is the
/// expected shape of this checkpoint or a real hole in it. Any other missing parameter is an error
/// whatever it says, and so is any tensor whose shape disagrees with the module the config built.
///
/// `n_heads` and `n_kv_heads` size the rotation permutation, and must be the ones the module was
/// built with.
pub fn load_safetensors(
    model: &mut Transformer,
    path: &Path,
    device: &Device,
    tied: bool,
    n_heads: usize,
    n_kv_heads: usize,
    d_model: usize,
) -> Result<OutputWeights, String> {
    let float_dtype: DType = device.settings().float_dtype.into();

    let mut store = SafetensorsStore::from_file(path)
        // The two adapters this file exists to warn about. Transpose first, cast second: the
        // transpose is a byte shuffle that works on any fixed-width dtype, so doing it in the
        // source's bf16 moves half the bytes.
        .with_from_adapter(PyTorchToBurnAdapter.chain(FloatCastAdapter::to(float_dtype)))
        // A tied checkpoint is legitimately short of `output.weight`, so a missing parameter has to
        // come back as a result to inspect rather than as an error. What is and is not acceptable
        // to be missing is decided below.
        .allow_partial(true);

    for (pattern, replacement) in remapping_rules() {
        store = store.with_key_remapping(pattern, replacement);
    }

    // The remapped keys are relative to the transformer, so the store is applied to it rather than
    // to the whole `Llama` — the same as the PyTorch path. Tensors in the file the module has no
    // slot for land in `unused` and are ignored: a checkpoint may carry precomputed rotary
    // frequencies or a trainer's leftovers, and neither is our business.
    let result = model
        .load_from(&mut store)
        .map_err(|err| format!("could not read {}: {err}", path.display()))?;

    if !result.errors.is_empty() {
        return Err(format!(
            "failed to apply {}: {:?}",
            path.display(),
            result.errors
        ));
    }

    let missing: Vec<&str> = result.missing.iter().map(|(p, _)| p.as_str()).collect();
    let output = match missing.as_slice() {
        [] => OutputWeights::FromCheckpoint,
        [TIED_OUTPUT] if tied => OutputWeights::TiedToEmbedding,
        [TIED_OUTPUT] => {
            return Err(format!(
                "{} carries no lm_head.weight and its config does not set \
                 tie_word_embeddings; refusing to guess which it meant",
                path.display()
            ))
        }
        _ => {
            return Err(format!(
                "{} is missing parameters the model has: {missing:?}. This usually means the \
                 checkpoint is sharded (a `model-00001-of-000NN.safetensors` set with a \
                 `model.safetensors.index.json`), which this loader does not read yet.",
                path.display()
            ))
        }
    };

    if output == OutputWeights::TiedToEmbedding {
        tie_output_to_embedding(model);
    }

    unpermute_rotary_projections(model, n_heads, n_kv_heads, d_model);

    Ok(output)
}

/// Checkpoint name to module path, applied in order to every key in the file.
///
/// These are `import.rs`'s HF-style rules. They are listed here rather than shared because the two
/// callers differ in what else they do (the PyTorch path is behind its own feature and carries the
/// Meta-style branch as well), and a table of regexes is clearer duplicated than abstracted.
fn remapping_rules() -> Vec<(&'static str, &'static str)> {
    vec![
        // lm_head.* -> output.*  (before the `model.` strip, since lm_head sits outside it)
        (r"lm_head\.(.+)", "output.$1"),
        // Everything else is under `model.`
        (r"model\.(.+)", "$1"),
        (r"embed_tokens\.(.+)", "tok_embeddings.$1"),
        (
            r"(layers\.[0-9]+)\.input_layernorm\.(.+)",
            "$1.attention_norm.$2",
        ),
        (
            r"(layers\.[0-9]+)\.post_attention_layernorm\.(.+)",
            "$1.ffn_norm.$2",
        ),
        (
            r"(layers\.[0-9]+)\.mlp\.down_proj\.(.+)",
            "$1.feed_forward.w2.$2",
        ),
        (
            r"(layers\.[0-9]+)\.mlp\.gate_proj\.(.+)",
            "$1.feed_forward.swiglu.linear_inner.$2",
        ),
        (
            r"(layers\.[0-9]+)\.mlp\.up_proj\.(.+)",
            "$1.feed_forward.swiglu.linear_outer.$2",
        ),
        (
            r"(layers\.[0-9]+)\.self_attn\.q_proj\.(.+)",
            "$1.attention.wq.$2",
        ),
        (
            r"(layers\.[0-9]+)\.self_attn\.k_proj\.(.+)",
            "$1.attention.wk.$2",
        ),
        (
            r"(layers\.[0-9]+)\.self_attn\.v_proj\.(.+)",
            "$1.attention.wv.$2",
        ),
        (
            r"(layers\.[0-9]+)\.self_attn\.o_proj\.(.+)",
            "$1.attention.wo.$2",
        ),
        // Every RMS norm's `weight` is our `gamma`. The adapter would find this on its own, but
        // stating it keeps the remapped names readable in an error message.
        (r"(.*)norm\.weight", "${1}norm.gamma"),
    ]
}

/// Materialize a tied output projection from the embedding matrix.
///
/// The embedding stores `[vocab, d_model]` and a burn linear wants `[d_in, d_out]`, so the tie is a
/// transpose. This copies rather than shares — burn modules own their parameters — so a tied model
/// costs the same memory as an untied one. That is the `TODO: tied weights` on `Transformer`, and
/// not something this change sets out to fix.
fn tie_output_to_embedding(model: &mut Transformer) {
    let embedding = model.tok_embeddings.weight.val();
    model.output.weight = burn::module::Param::from_tensor(embedding.swap_dims(0, 1));
}

/// Undo Hugging Face's query/key permutation so the weights match burn's rotation convention.
///
/// `transformers`' conversion script reorders each head's output channels of `q_proj` and `k_proj`
/// so that rotating the two halves of a head is equivalent to rotating adjacent pairs. burn rotates
/// adjacent pairs, so the reordering has to come back out. Both weights are already in burn's
/// `[d_in, d_out]` layout by the time this runs, so the head structure lives in the last dimension:
/// split it into `(heads, 2, head_dim/2)` and swap the trailing two axes back to
/// `(heads, head_dim/2, 2)`.
fn unpermute_rotary_projections(
    model: &mut Transformer,
    n_heads: usize,
    n_kv_heads: usize,
    d_model: usize,
) {
    let kv_dim = d_model * n_kv_heads / n_heads;
    let permute = |w: Tensor<2>, heads: usize, d_in: usize, d_out: usize| {
        w.reshape([d_in, heads, 2, d_out / heads / 2])
            .swap_dims(2, 3)
            .reshape([d_in, d_out])
    };

    let layers = core::mem::take(&mut model.layers);
    model.layers = layers
        .into_iter()
        .map(|mut layer| {
            layer.attention.wq.weight = layer
                .attention
                .wq
                .weight
                .map(|w| permute(w, n_heads, d_model, d_model));
            layer.attention.wk.weight = layer
                .attention
                .wk
                .weight
                .map(|w| permute(w, n_kv_heads, d_model, kv_dim));
            layer
        })
        .collect();
}

#[cfg(test)]
mod tests {
    use burn_store::KeyRemapper;

    use super::*;

    /// Run a checkpoint key through the rule set the way the store does.
    fn remap(key: &str) -> String {
        let remapper = remapping_rules().into_iter().fold(
            KeyRemapper::new(),
            |remapper, (pattern, replacement)| {
                remapper.add_pattern(pattern, replacement).expect("regex")
            },
        );
        let tensor = burn_store::bridge::from_data(
            burn::tensor::TensorData::new(vec![0.0f32], [1]),
            key.to_string(),
            None,
        );
        remapper.remap(vec![tensor]).0[0].name.clone()
    }

    /// Every name a canonical Llama safetensors file carries, mapped onto the module path it has to
    /// land on. This is the table that decides whether the weights end up in the right layers, so
    /// it is checked name by name rather than by loading a file.
    #[test]
    fn every_checkpoint_name_maps_onto_its_module_path() {
        for (checkpoint, module) in [
            ("model.embed_tokens.weight", "tok_embeddings.weight"),
            ("lm_head.weight", "output.weight"),
            ("model.norm.weight", "norm.gamma"),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "layers.0.attention.wq.weight",
            ),
            (
                "model.layers.7.self_attn.k_proj.weight",
                "layers.7.attention.wk.weight",
            ),
            (
                "model.layers.7.self_attn.v_proj.weight",
                "layers.7.attention.wv.weight",
            ),
            (
                "model.layers.15.self_attn.o_proj.weight",
                "layers.15.attention.wo.weight",
            ),
            (
                "model.layers.3.mlp.gate_proj.weight",
                "layers.3.feed_forward.swiglu.linear_inner.weight",
            ),
            (
                "model.layers.3.mlp.up_proj.weight",
                "layers.3.feed_forward.swiglu.linear_outer.weight",
            ),
            (
                "model.layers.3.mlp.down_proj.weight",
                "layers.3.feed_forward.w2.weight",
            ),
            (
                "model.layers.3.input_layernorm.weight",
                "layers.3.attention_norm.gamma",
            ),
            (
                "model.layers.3.post_attention_layernorm.weight",
                "layers.3.ffn_norm.gamma",
            ),
        ] {
            assert_eq!(remap(checkpoint), module, "remapping {checkpoint}");
        }
    }

    /// What the permutation must actually do, stated as the channel mapping rather than as a
    /// reshape.
    ///
    /// A Hugging Face head lays its rotated channels out as two halves — the `hd/2` real parts and
    /// then the `hd/2` imaginary parts — because its rotation pairs a channel with the one `hd/2`
    /// along. burn pairs adjacent channels, so it wants them interleaved: `r0 i0 r1 i1 …`. So
    /// output channel `2k` must come from input channel `k`, and output channel `2k+1` from input
    /// channel `hd/2 + k`, per head. A head width of 6 is used rather than 4 because at `hd == 4`
    /// the mapping happens to be its own inverse and a swapped reshape would pass anyway.
    #[test]
    fn the_rotary_permutation_interleaves_each_heads_halves() {
        let device = Device::default();
        let (heads, head_dim, d_in) = (2usize, 6usize, 2usize);
        let d_out = heads * head_dim;
        let half = head_dim / 2;

        // Each element is its own flat index, so the mapping can be read straight off the result.
        let values: Vec<f32> = (0..(d_in * d_out) as i32).map(|v| v as f32).collect();
        let original = Tensor::<2>::from_data(
            burn::tensor::TensorData::new(values, [d_in, d_out]),
            &device,
        );

        let permuted = original
            .clone()
            .reshape([d_in, heads, 2, d_out / heads / 2])
            .swap_dims(2, 3)
            .reshape([d_in, d_out])
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();

        for row in 0..d_in {
            for head in 0..heads {
                for k in 0..half {
                    let base = row * d_out + head * head_dim;
                    assert_eq!(
                        permuted[base + 2 * k],
                        (base + k) as f32,
                        "real part {k} of head {head}"
                    );
                    assert_eq!(
                        permuted[base + 2 * k + 1],
                        (base + half + k) as f32,
                        "imaginary part {k} of head {head}"
                    );
                }
            }
        }
    }
}
