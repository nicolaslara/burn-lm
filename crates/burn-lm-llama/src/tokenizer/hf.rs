//! The tokenizer a canonical Hugging Face repo actually ships.
//!
//! `Tiktoken` reads Meta's `tokenizer.model` — a plain text file of base64 token bytes and their
//! merge ranks — and hardcodes Llama-3's eleven named special tokens plus the 245 reserved ones,
//! because that file carries no special tokens at all. Canonical repos do not ship it. They ship
//! `tokenizer.json`, the `tokenizers` crate's own format, which carries the vocabulary, the merges,
//! the pre-tokenizer, and every added token with its id. Reading that is both less code and less
//! guessing: the special token ids come from the file instead of being reconstructed from a
//! convention.
//!
//! So this is the tokenizer for the direct-loading path, and `Tiktoken` stays exactly as it is for
//! the published checkpoints, whose downloads are Meta's file.
//!
//! The one piece that needs care is `decode_bytes`. Generation streams token by token, and a
//! byte-level BPE vocabulary routinely splits a multi-byte character across two tokens, so decoding
//! one token at a time has to yield that token's *bytes* and let the caller reassemble them —
//! `Tiktoken` overrides this for the same reason. `tokenizers`' own `decode` would hand back a
//! replacement character for a half character, silently corrupting any output containing an emoji
//! or an accent. The fix is to read the bytes out of the token string directly, undoing the
//! byte-to-printable-character mapping that byte-level BPE stores its vocabulary in.

use tokenizers::Tokenizer as BaseTokenizer;

use super::Tokenizer;

/// Llama-3's beginning-of-text token, and the Llama-2/TinyLlama spelling as a fallback.
const BOS_CANDIDATES: [&str; 2] = ["<|begin_of_text|>", "<s>"];
/// End of text. Not end of *turn* — see `stop_ids`.
const EOS_CANDIDATES: [&str; 2] = ["<|end_of_text|>", "</s>"];
/// The tokens that end a generation. An instruct model stops a reply with end-of-turn, not
/// end-of-text, so a generation that only watched for `eos` would run to the token limit every
/// time. These are the same three `Tiktoken::stop_ids` reports.
const STOP_CANDIDATES: [&str; 4] = ["<|end_of_text|>", "<|eot_id|>", "<|eom_id|>", "</s>"];

#[derive(Debug, Clone)]
pub struct HfTokenizer {
    bpe: BaseTokenizer,
    bos_token_id: u32,
    eos_token_id: u32,
    stop_token_ids: Vec<u32>,
}

impl Tokenizer for HfTokenizer {
    /// Load a `tokenizer.json`.
    fn new(tokenizer_path: &str) -> Result<Self, String> {
        let bpe = BaseTokenizer::from_file(tokenizer_path)
            .map_err(|err| format!("could not read the tokenizer at {tokenizer_path}: {err}"))?;

        let id_of =
            |names: &[&str]| -> Option<u32> { names.iter().find_map(|name| bpe.token_to_id(name)) };

        let bos_token_id = id_of(&BOS_CANDIDATES).ok_or_else(|| {
            format!(
                "{tokenizer_path} has none of the beginning-of-text tokens \
                 {BOS_CANDIDATES:?}; this loader expects a Llama-family tokenizer"
            )
        })?;
        let eos_token_id = id_of(&EOS_CANDIDATES).ok_or_else(|| {
            format!(
                "{tokenizer_path} has none of the end-of-text tokens {EOS_CANDIDATES:?}; \
                 this loader expects a Llama-family tokenizer"
            )
        })?;
        // Whichever of the stop tokens this vocabulary actually has. A base (non-instruct) model
        // has no end-of-turn token and stops on end-of-text alone, which is correct for it.
        let stop_token_ids = STOP_CANDIDATES
            .iter()
            .filter_map(|name| bpe.token_to_id(name))
            .collect();

        Ok(Self {
            bpe,
            bos_token_id,
            eos_token_id,
            stop_token_ids,
        })
    }

    /// Encode `text`, honoring any special tokens written into it.
    ///
    /// The `false` passed to `tokenizers` turns off the post-processor, which is what would add a
    /// BOS of its own; `bos`/`eos` here are the caller's explicit request, matching `Tiktoken`.
    /// Special tokens spelled out in the text (`<|start_header_id|>`, as the chat framing writes
    /// them) are still recognised, because added tokens are matched before pre-tokenization
    /// regardless of that flag.
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if bos {
            tokens.push(self.bos_token_id);
        }
        tokens.extend(
            self.bpe
                .encode(text, false)
                .expect("should encode the prompt")
                .get_ids()
                .iter()
                .copied(),
        );
        if eos {
            tokens.push(self.eos_token_id);
        }
        tokens
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.bpe
            .decode(tokens, false)
            .expect("should decode tokens")
    }

    /// The bytes `tokens` stand for, which may not be valid UTF-8 on their own.
    ///
    /// A byte-level BPE vocabulary stores each token as the printable characters that byte-level
    /// pre-tokenization maps its bytes to, so the bytes come back by inverting that map on the
    /// token's own string. An added token (`<|eot_id|>`) is not byte-encoded — its string is the
    /// literal text — which is what the fallback covers.
    fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &token in tokens {
            match self.bpe.id_to_token(token) {
                Some(text) => bytes.extend(byte_level_to_bytes(&text)),
                // An id outside the vocabulary contributes nothing rather than panicking
                // mid-generation.
                None => continue,
            }
        }
        bytes
    }

    fn bos_id(&self) -> u32 {
        self.bos_token_id
    }

    fn eos_id(&self) -> u32 {
        self.eos_token_id
    }

    fn stop_ids(&self) -> Vec<u32> {
        self.stop_token_ids.clone()
    }
}

/// Undo byte-level BPE's byte-to-character mapping.
///
/// GPT-2's byte-level pre-tokenizer, which every Llama-3 tokenizer uses, rewrites each of the 256
/// bytes as a single printable Unicode character so that a vocabulary is plain text: the 188 bytes
/// that are already printable keep their own character, and the other 68 are moved up into
/// U+0100..U+0143. Mapping those characters back gives the token's bytes exactly. A string with a
/// character outside that alphabet is not byte-encoded at all — an added token such as
/// `<|eot_id|>` — so its own UTF-8 is the answer.
fn byte_level_to_bytes(token: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(token.len());
    for ch in token.chars() {
        match byte_of(ch) {
            Some(byte) => bytes.push(byte),
            None => return token.as_bytes().to_vec(),
        }
    }
    bytes
}

/// Whether byte-level encoding leaves a byte as its own character: `'!'..='~'`, `'¡'..='¬'`,
/// `'®'..='ÿ'` — 188 of the 256.
fn is_printable_byte(code: u32) -> bool {
    (0x21..=0x7E).contains(&code) || (0xA1..=0xAC).contains(&code) || (0xAE..=0xFF).contains(&code)
}

/// How many bytes had to be moved, and so how far past U+0100 the alphabet runs.
const MOVED: u32 = 68;

/// The byte one byte-level character stands for.
///
/// The 68 bytes that are not printable were assigned U+0100 onwards *in ascending byte order*, and
/// they fall in three contiguous runs — the control bytes and space (`0x00..=0x20`), the run from
/// DEL through `0xA0`, and the lone soft hyphen `0xAD`. So the character's offset maps back to a
/// byte by arithmetic, with no table to build on every character decoded. The test below checks
/// this arithmetic against the ascending-order definition it comes from.
fn byte_of(ch: char) -> Option<u8> {
    let code = ch as u32;
    if is_printable_byte(code) {
        return Some(code as u8);
    }
    if !(0x100..0x100 + MOVED).contains(&code) {
        return None;
    }
    let index = code - 0x100;
    Some(match index {
        // 0x00..=0x20: 33 bytes, in order.
        0..=32 => index as u8,
        // 0x7F..=0xA0: 34 bytes, starting at index 33.
        33..=66 => (0x7F + (index - 33)) as u8,
        // 0xAD, the soft hyphen, alone at the end.
        _ => 0xAD,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes byte-level encoding had to move, straight from the definition: every byte that is
    /// not printable, in ascending order. `byte_of`'s arithmetic has to agree with this.
    fn remapped_bytes() -> Vec<u8> {
        (0u32..=255)
            .filter(|&byte| !is_printable_byte(byte))
            .map(|byte| byte as u8)
            .collect()
    }

    /// The mapping is a bijection over all 256 bytes: every byte encodes to exactly one character
    /// and decodes back to itself. Checked directly, since the whole of `decode_bytes` rests on it.
    #[test]
    fn every_byte_round_trips_through_the_byte_level_alphabet() {
        let remapped = remapped_bytes();
        assert_eq!(
            remapped.len() as u32,
            MOVED,
            "the number of non-printable bytes decides how far the alphabet runs"
        );

        for byte in 0u8..=255 {
            let ch = match remapped.iter().position(|&b| b == byte) {
                Some(index) => char::from_u32(0x100 + index as u32).unwrap(),
                None => char::from_u32(byte as u32).unwrap(),
            };
            assert_eq!(
                byte_of(ch),
                Some(byte),
                "byte {byte} encoded as {ch:?} did not come back"
            );
        }
    }

    /// A space is byte 0x20, which is not printable, and byte-level BPE writes it as 'Ġ' — the
    /// single most common character in a Llama vocabulary. If this is wrong, every word boundary in
    /// every streamed reply is wrong.
    #[test]
    fn the_leading_space_marker_decodes_to_a_space() {
        assert_eq!(byte_level_to_bytes("Ġthe"), b" the".to_vec());
    }

    /// An added token is literal text, not byte-encoded, and has to survive as itself.
    #[test]
    fn an_added_token_decodes_to_its_own_text() {
        assert_eq!(byte_level_to_bytes("<|eot_id|>"), b"<|eot_id|>".to_vec());
    }

    /// A multi-byte character's bytes are spelled out one byte-level character each, and must
    /// reassemble into the character.
    #[test]
    fn a_multibyte_character_reassembles_from_its_bytes() {
        // '€' is E2 82 AC. E2 and 82 are among the moved bytes; AC is printable ('¬').
        let remapped = remapped_bytes();
        let encoded: String = "€"
            .as_bytes()
            .iter()
            .map(|&byte| match remapped.iter().position(|&b| b == byte) {
                Some(index) => char::from_u32(0x100 + index as u32).unwrap(),
                None => char::from_u32(byte as u32).unwrap(),
            })
            .collect();
        assert_eq!(byte_level_to_bytes(&encoded), "€".as_bytes().to_vec());
    }
}
