/// The seed used by spaCy's `hash_string`.
pub const SPACY_STRING_SEED: u64 = 1;

const MURMUR64_MULTIPLIER: u64 = 0xc6a4_a793_5bd1_e995;
const MURMUR64_ROTATION: u32 = 47;

/// Hash bytes with the `MurmurHash64A` variant used by spaCy's `murmurhash`
/// package.
///
/// The implementation reads blocks as little-endian values so its result is
/// deterministic on every Rust target that we support.
#[must_use]
pub fn hash_bytes(bytes: &[u8]) -> u64 {
    murmur_hash_64a(bytes, SPACY_STRING_SEED)
}

/// Hash a UTF-8 string exactly as spaCy's `spacy.strings.hash_string` does.
#[must_use]
pub fn hash_string(text: &str) -> u64 {
    hash_bytes(text.as_bytes())
}

fn murmur_hash_64a(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = seed ^ (bytes.len() as u64).wrapping_mul(MURMUR64_MULTIPLIER);

    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let mut block = [0_u8; 8];
        block.copy_from_slice(chunk);
        let mut value = u64::from_le_bytes(block);
        value = value.wrapping_mul(MURMUR64_MULTIPLIER);
        value ^= value >> MURMUR64_ROTATION;
        value = value.wrapping_mul(MURMUR64_MULTIPLIER);

        hash ^= value;
        hash = hash.wrapping_mul(MURMUR64_MULTIPLIER);
    }

    let remainder = chunks.remainder();
    if !remainder.is_empty() {
        let mut tail = 0_u64;
        for (shift, byte) in remainder.iter().enumerate() {
            tail ^= u64::from(*byte) << (shift * 8);
        }
        hash ^= tail;
        hash = hash.wrapping_mul(MURMUR64_MULTIPLIER);
    }

    hash ^= hash >> MURMUR64_ROTATION;
    hash = hash.wrapping_mul(MURMUR64_MULTIPLIER);
    hash ^= hash >> MURMUR64_ROTATION;
    hash
}

#[cfg(test)]
mod tests {
    use super::{hash_bytes, hash_string};

    #[test]
    fn hashes_utf8_bytes_without_normalizing() {
        assert_eq!(hash_string("é"), hash_bytes("é".as_bytes()));
        assert_ne!(hash_string("é"), hash_string("e\u{301}"));
    }

    #[test]
    fn changing_any_input_byte_changes_the_fixture_hash() {
        assert_ne!(hash_string("spaCy"), hash_string("spacy"));
    }

    #[test]
    fn matches_spacy_3_8_golden_hashes() {
        let fixtures = [
            ("", 14_313_749_767_032_693_980),
            ("hello", 5_983_625_672_228_268_878),
            ("spaCy", 6_772_933_960_739_496_234),
            ("é", 16_256_712_947_286_012_828),
            ("e\u{301}", 12_745_818_603_700_845_520),
            ("日本語", 1_998_301_522_555_383_300),
            ("🦀", 814_917_477_307_458_328),
        ];

        for (text, expected) in fixtures {
            assert_eq!(hash_string(text), expected, "hash mismatch for {text:?}");
        }
    }
}
