//! Reader for the `.mpk` weight files burn used to write.
//!
//! The published Llama checkpoints on Hugging Face (`tracel-ai/*-burn`, see `pretrained.rs`) were
//! saved with burn's `NamedMpkFileRecorder<HalfPrecisionSettings>`: one msgpack document that
//! mirrors the module tree, floats stored as f16 and ints as i16. burn `main` dropped that whole
//! recorder family with the record refactor (tracel-ai/burn#5083) in favor of `ModuleRecord` and
//! the burnpack format, and offers no reader for the old files. Nothing upstream re-published the
//! weights, so this module keeps them loadable.
//!
//! The on-disk shape is simple. A module is a map keyed by its field names; a `Vec` of modules is
//! an array; `Option::None` and constant fields (`usize`, `f64`) are nil or scalars; and every
//! parameter is a two-key map `{ "id": "<param id>", "param": <TensorData> }`. `TensorData` itself
//! (`bytes`, `shape`, `dtype`) still serializes exactly the same way in today's burn, so its own
//! deserializer reads the leaves. We stream the document once, keep only the parameter leaves under
//! their dotted path (`layers.0.attention.wq.weight`), and hand them to burn-store's applier, which
//! places tensors by that same path.
//!
//! Dtype handling copies the old recorder: a float parameter is converted to the device's default
//! float dtype and an int parameter to its default int dtype, so a model loaded from an f16 file on
//! an f32 device is f32 in memory, as before. Quantized tensors are refused rather than mis-read —
//! the quantization scheme's serialized layout changed with the same upgrade, and serde would
//! otherwise fill the new fields with defaults and load a scheme the file never described.

use std::{fs::File, io::BufReader, path::Path};

use burn::{
    module::Module,
    tensor::{DType, Device, TensorData},
};
use burn_store::{bridge, ModuleSnapshot, PathFilter};
use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};

/// Read the legacy `.mpk` file at `path` and apply its parameters to `module` on `device`.
///
/// Fails on a malformed file, a parameter the module does not have (or vice versa), a shape
/// mismatch, or quantized weights.
pub fn load_into<M: Module>(module: &mut M, path: &Path, device: &Device) -> Result<(), String> {
    let file = File::open(path)
        .map_err(|err| format!("could not open {}: {err}", path.display()))?;
    let mut reader = rmp_serde::Deserializer::new(BufReader::new(file));

    let mut walk = Walk {
        path: Vec::new(),
        tensors: Vec::new(),
        device: device.clone(),
    };
    NodeSeed(&mut walk)
        .deserialize(&mut reader)
        .map_err(|err| format!("could not read {}: {err}", path.display()))?;

    let result = module.apply(walk.tensors, None::<PathFilter>, None, false);
    if !result.errors.is_empty() {
        return Err(format!("failed to apply {}: {:?}", path.display(), result.errors));
    }
    if !result.missing.is_empty() {
        let missing: Vec<&str> = result.missing.iter().map(|(p, _)| p.as_str()).collect();
        return Err(format!(
            "{} is missing parameters the model has: {missing:?}",
            path.display()
        ));
    }
    if !result.unused.is_empty() {
        return Err(format!(
            "{} holds parameters the model does not have: {:?}",
            path.display(),
            result.unused
        ));
    }
    Ok(())
}

/// State carried down the document: the dotted path to the node being read and the parameter
/// leaves collected so far.
struct Walk {
    path: Vec<String>,
    tensors: Vec<burn_store::burn_pack::Tensor>,
    device: Device,
}

impl Walk {
    /// Record one parameter leaf at the current path, converting its dtype the way the old
    /// recorder did on load.
    fn leaf<E: de::Error>(&mut self, id: String, data: TensorData) -> Result<(), E> {
        let name = self.path.join(".");
        let settings = self.device.settings();
        let data = match data.dtype {
            DType::QFloat(_) => {
                return Err(E::custom(format!(
                    "parameter `{name}` is quantized; quantized .mpk files predate the current \
                     quantization scheme and cannot be read"
                )))
            }
            DType::Bool(_) => data,
            DType::F64 | DType::F32 | DType::Flex32 | DType::F16 | DType::BF16 => {
                data.convert_dtype(settings.float_dtype.into())
            }
            DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8 => data.convert_dtype(settings.int_dtype.into()),
        };
        // The old recorder wrote `ParamId` as its decimal value; an id we cannot parse is simply
        // not carried over, and the applier keeps the module's own.
        let param_id = id.parse::<u64>().ok();
        self.tensors.push(bridge::from_data(data, name, param_id));
        Ok(())
    }
}

/// One node of the document. Maps and arrays recurse; a `{id, param}` map is a parameter leaf;
/// anything else (nil for `None`, scalars for constants) is skipped.
struct NodeSeed<'a>(&'a mut Walk);

impl<'de, 'a> DeserializeSeed<'de> for NodeSeed<'a> {
    type Value = ();

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }
}

impl<'de, 'a> Visitor<'de> for NodeSeed<'a> {
    type Value = ();

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a burn module record node")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let Some(first) = map.next_key::<String>()? else {
            return Ok(());
        };

        if first == "id" {
            let id: String = map.next_value()?;
            match map.next_key::<String>()?.as_deref() {
                Some("param") => {}
                other => {
                    return Err(de::Error::custom(format!(
                        "expected `param` after `id` at `{}`, found {other:?}",
                        self.0.path.join(".")
                    )))
                }
            }
            let data: TensorData = map.next_value()?;
            if map.next_key::<String>()?.is_some() {
                return Err(de::Error::custom(format!(
                    "unexpected extra key in parameter `{}`",
                    self.0.path.join(".")
                )));
            }
            return self.0.leaf(id, data);
        }

        let mut key = first;
        loop {
            self.0.path.push(key);
            map.next_value_seed(NodeSeed(self.0))?;
            self.0.path.pop();
            match map.next_key::<String>()? {
                Some(next) => key = next,
                None => return Ok(()),
            }
        }
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let mut index = 0usize;
        loop {
            self.0.path.push(index.to_string());
            let more = seq.next_element_seed(NodeSeed(self.0))?.is_some();
            self.0.path.pop();
            if !more {
                return Ok(());
            }
            index += 1;
        }
    }

    // Everything below is a constant or an absent optional field: nothing to load.

    fn visit_unit<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_none<E: de::Error>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_any(self)
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E: de::Error>(self, _: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_bytes<E: de::Error>(self, _: &[u8]) -> Result<(), E> {
        Ok(())
    }
}
