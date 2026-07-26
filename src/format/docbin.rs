use std::collections::BTreeMap;
use std::io::{Read, Write};

use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use thiserror::Error;

/// Resource limits applied while decoding a spaCy `DocBin`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DocBinLimits {
    pub max_compressed_bytes: usize,
    pub max_decompressed_bytes: usize,
    pub max_documents: usize,
    pub max_tokens: usize,
    pub max_attributes: usize,
    pub max_strings: usize,
    pub max_metadata_bytes_per_document: usize,
}

impl Default for DocBinLimits {
    fn default() -> Self {
        Self {
            max_compressed_bytes: 64 * 1024 * 1024,
            max_decompressed_bytes: 128 * 1024 * 1024,
            max_documents: 100_000,
            max_tokens: 1_000_000,
            max_attributes: 256,
            max_strings: 250_000,
            max_metadata_bytes_per_document: 16 * 1024 * 1024,
        }
    }
}

/// `DocBin` resource guarded by [`DocBinLimits`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocBinLimitResource {
    CompressedBytes,
    DecompressedBytes,
    Documents,
    Tokens,
    Attributes,
    Strings,
    DocumentMetadataBytes,
}

impl DocBinLimitResource {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompressedBytes => "compressed_bytes",
            Self::DecompressedBytes => "decompressed_bytes",
            Self::Documents => "documents",
            Self::Tokens => "tokens",
            Self::Attributes => "attributes",
            Self::Strings => "strings",
            Self::DocumentMetadataBytes => "document_metadata_bytes",
        }
    }
}

impl std::fmt::Display for DocBinLimitResource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DocBinMessage {
    version: String,
    attrs: Vec<u64>,
    #[serde(with = "serde_bytes")]
    tokens: Vec<u8>,
    #[serde(with = "serde_bytes")]
    spaces: Vec<u8>,
    #[serde(with = "serde_bytes")]
    lengths: Vec<u8>,
    strings: Vec<String>,
    cats: Vec<BTreeMap<String, f64>>,
    flags: Vec<BTreeMap<String, bool>>,
    span_groups: Vec<ByteBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    user_data: Vec<Option<ByteBuf>>,
}

/// One semantically decoded document from a spaCy `DocBin`.
#[derive(Clone, Debug, PartialEq)]
pub struct DocRecord {
    pub tokens: Vec<Vec<u64>>,
    pub spaces: Vec<bool>,
    pub cats: BTreeMap<String, f64>,
    pub flags: BTreeMap<String, bool>,
    pub span_groups: Vec<u8>,
    pub user_data: Option<Vec<u8>>,
}

/// A decoded spaCy `DocBin`.
#[derive(Clone, Debug, PartialEq)]
pub struct DocBin {
    version: String,
    attrs: Vec<u64>,
    strings: Vec<String>,
    records: Vec<DocRecord>,
}

#[derive(Debug, Error)]
pub enum DocBinError {
    #[error("DocBin zlib stream is invalid: {0}")]
    Compression(#[from] std::io::Error),
    #[error("DocBin msgpack payload is invalid: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("DocBin could not be encoded: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("DocBin attribute list must not be empty")]
    NoAttributes,
    #[error("{field} byte length {actual} is not divisible by element width {width}")]
    MisalignedBytes {
        field: &'static str,
        actual: usize,
        width: usize,
    },
    #[error("DocBin token matrix has {actual} values, but lengths and attrs require {expected}")]
    TokenValueCount { expected: usize, actual: usize },
    #[error("DocBin spaces have {actual} values, but documents require {expected}")]
    SpaceCount { expected: usize, actual: usize },
    #[error("DocBin metadata field {field} has {actual} entries, expected {expected}")]
    MetadataCount {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("DocBin length at index {index} is negative: {value}")]
    NegativeLength { index: usize, value: i32 },
    #[error("DocBin token count overflow")]
    TokenCountOverflow,
    #[error("DocBin resource {resource} has {actual} units, exceeding limit {limit}")]
    LimitExceeded {
        resource: DocBinLimitResource,
        actual: usize,
        limit: usize,
    },
}

impl DocBin {
    /// Decode a zlib-compressed spaCy `DocBin`.
    ///
    /// # Errors
    ///
    /// Returns [`DocBinError`] if decompression, msgpack decoding, or any
    /// structural validation fails.
    pub fn from_bytes(compressed: &[u8]) -> Result<Self, DocBinError> {
        Self::from_bytes_with_limits(compressed, &DocBinLimits::default())
    }

    /// Decode a zlib-compressed spaCy `DocBin` with caller-selected limits.
    ///
    /// # Errors
    ///
    /// Returns [`DocBinError`] before msgpack decoding when compressed or
    /// decompressed bytes exceed their configured limits. Decoded document,
    /// token, attribute, string, and metadata counts are also bounded.
    pub fn from_bytes_with_limits(
        compressed: &[u8],
        limits: &DocBinLimits,
    ) -> Result<Self, DocBinError> {
        check_limit(
            DocBinLimitResource::CompressedBytes,
            compressed.len(),
            limits.max_compressed_bytes,
        )?;
        let mut decoder = ZlibDecoder::new(compressed);
        let mut payload = Vec::new();
        decoder
            .by_ref()
            .take(
                u64::try_from(limits.max_decompressed_bytes)
                    .unwrap_or(u64::MAX)
                    .saturating_add(1),
            )
            .read_to_end(&mut payload)?;
        check_limit(
            DocBinLimitResource::DecompressedBytes,
            payload.len(),
            limits.max_decompressed_bytes,
        )?;
        let message: DocBinMessage = rmp_serde::from_slice(&payload)?;
        Self::from_message(message, limits)
    }

    /// Encode this container as a spaCy-compatible zlib-compressed msgpack
    /// payload.
    ///
    /// # Errors
    ///
    /// Returns [`DocBinError`] if the in-memory records are inconsistent or
    /// serialization fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DocBinError> {
        let message = self.to_message()?;
        let payload = rmp_serde::to_vec_named(&message)?;
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&payload)?;
        Ok(encoder.finish()?)
    }

    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    #[must_use]
    pub fn attrs(&self) -> &[u64] {
        &self.attrs
    }

    #[must_use]
    pub fn strings(&self) -> &[String] {
        &self.strings
    }

    #[must_use]
    pub fn records(&self) -> &[DocRecord] {
        &self.records
    }

    fn from_message(message: DocBinMessage, limits: &DocBinLimits) -> Result<Self, DocBinError> {
        if message.attrs.is_empty() {
            return Err(DocBinError::NoAttributes);
        }
        check_limit(
            DocBinLimitResource::Attributes,
            message.attrs.len(),
            limits.max_attributes,
        )?;
        check_limit(
            DocBinLimitResource::Strings,
            message.strings.len(),
            limits.max_strings,
        )?;
        ensure_aligned("tokens", message.tokens.len(), 8)?;
        ensure_aligned("lengths", message.lengths.len(), 4)?;
        check_limit(
            DocBinLimitResource::Documents,
            message.lengths.len() / 4,
            limits.max_documents,
        )?;

        let lengths = decode_i32_le(&message.lengths)
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                usize::try_from(value).map_err(|_| DocBinError::NegativeLength { index, value })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let document_count = lengths.len();
        ensure_metadata_count("cats", document_count, message.cats.len())?;
        ensure_metadata_count("flags", document_count, message.flags.len())?;
        ensure_metadata_count("span_groups", document_count, message.span_groups.len())?;
        if !message.user_data.is_empty() {
            ensure_metadata_count("user_data", document_count, message.user_data.len())?;
        }

        let token_count = lengths.iter().try_fold(0_usize, |total, length| {
            total
                .checked_add(*length)
                .ok_or(DocBinError::TokenCountOverflow)
        })?;
        check_limit(DocBinLimitResource::Tokens, token_count, limits.max_tokens)?;
        let expected_values = token_count
            .checked_mul(message.attrs.len())
            .ok_or(DocBinError::TokenCountOverflow)?;
        let token_values = decode_u64_le(&message.tokens);
        if token_values.len() != expected_values {
            return Err(DocBinError::TokenValueCount {
                expected: expected_values,
                actual: token_values.len(),
            });
        }
        if message.spaces.len() != token_count {
            return Err(DocBinError::SpaceCount {
                expected: token_count,
                actual: message.spaces.len(),
            });
        }

        let mut value_offset = 0;
        let mut token_offset = 0;
        let mut cats = message.cats.into_iter();
        let mut flags = message.flags.into_iter();
        let mut span_groups = message.span_groups.into_iter();
        let mut user_data = message.user_data.into_iter();
        let mut records = Vec::with_capacity(document_count);

        for length in lengths {
            let mut tokens = Vec::with_capacity(length);
            for _ in 0..length {
                let end = value_offset + message.attrs.len();
                tokens.push(token_values[value_offset..end].to_vec());
                value_offset = end;
            }
            let space_end = token_offset + length;
            let spaces = message.spaces[token_offset..space_end]
                .iter()
                .map(|value| *value != 0)
                .collect();
            token_offset = space_end;

            let span_groups = span_groups
                .next()
                .expect("metadata counts were validated")
                .into_vec();
            let user_data = user_data.next().flatten().map(ByteBuf::into_vec);
            let metadata_bytes = span_groups
                .len()
                .checked_add(user_data.as_ref().map_or(0, Vec::len))
                .ok_or(DocBinError::TokenCountOverflow)?;
            check_limit(
                DocBinLimitResource::DocumentMetadataBytes,
                metadata_bytes,
                limits.max_metadata_bytes_per_document,
            )?;

            records.push(DocRecord {
                tokens,
                spaces,
                cats: cats.next().expect("metadata counts were validated"),
                flags: flags.next().expect("metadata counts were validated"),
                span_groups,
                user_data,
            });
        }

        Ok(Self {
            version: message.version,
            attrs: message.attrs,
            strings: message.strings,
            records,
        })
    }

    fn to_message(&self) -> Result<DocBinMessage, DocBinError> {
        if self.attrs.is_empty() {
            return Err(DocBinError::NoAttributes);
        }

        let mut token_values = Vec::new();
        let mut spaces = Vec::new();
        let mut lengths = Vec::with_capacity(self.records.len());
        let mut cats = Vec::with_capacity(self.records.len());
        let mut flags = Vec::with_capacity(self.records.len());
        let mut span_groups = Vec::with_capacity(self.records.len());
        let has_user_data = self.records.iter().any(|record| record.user_data.is_some());
        let mut user_data = if has_user_data {
            Vec::with_capacity(self.records.len())
        } else {
            Vec::new()
        };

        for record in &self.records {
            let length =
                i32::try_from(record.tokens.len()).map_err(|_| DocBinError::TokenCountOverflow)?;
            lengths.push(length);
            if record.spaces.len() != record.tokens.len() {
                return Err(DocBinError::SpaceCount {
                    expected: record.tokens.len(),
                    actual: record.spaces.len(),
                });
            }
            for token in &record.tokens {
                if token.len() != self.attrs.len() {
                    return Err(DocBinError::TokenValueCount {
                        expected: self.attrs.len(),
                        actual: token.len(),
                    });
                }
                token_values.extend(token);
            }
            spaces.extend(record.spaces.iter().map(|space| u8::from(*space)));
            cats.push(record.cats.clone());
            flags.push(record.flags.clone());
            span_groups.push(ByteBuf::from(record.span_groups.clone()));
            if has_user_data {
                user_data.push(record.user_data.clone().map(ByteBuf::from));
            }
        }

        Ok(DocBinMessage {
            version: self.version.clone(),
            attrs: self.attrs.clone(),
            tokens: encode_u64_le(&token_values),
            spaces,
            lengths: encode_i32_le(&lengths),
            strings: self.strings.clone(),
            cats,
            flags,
            span_groups,
            user_data,
        })
    }
}

fn check_limit(
    resource: DocBinLimitResource,
    actual: usize,
    limit: usize,
) -> Result<(), DocBinError> {
    if actual > limit {
        Err(DocBinError::LimitExceeded {
            resource,
            actual,
            limit,
        })
    } else {
        Ok(())
    }
}

fn ensure_aligned(field: &'static str, actual: usize, width: usize) -> Result<(), DocBinError> {
    if actual.is_multiple_of(width) {
        Ok(())
    } else {
        Err(DocBinError::MisalignedBytes {
            field,
            actual,
            width,
        })
    }
}

fn ensure_metadata_count(
    field: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), DocBinError> {
    if actual == expected {
        Ok(())
    } else {
        Err(DocBinError::MetadataCount {
            field,
            expected,
            actual,
        })
    }
}

fn decode_u64_le(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut value = [0_u8; 8];
            value.copy_from_slice(chunk);
            u64::from_le_bytes(value)
        })
        .collect()
}

fn encode_u64_le(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode_i32_le(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            let mut value = [0_u8; 4];
            value.copy_from_slice(chunk);
            i32::from_le_bytes(value)
        })
        .collect()
}

fn encode_i32_le(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{DocBin, DocBinError, DocBinLimitResource, DocBinLimits, DocRecord};

    fn fixture() -> DocBin {
        DocBin {
            version: "0.1".to_owned(),
            attrs: vec![65],
            strings: vec!["alpha".to_owned()],
            records: vec![DocRecord {
                tokens: vec![vec![1], vec![2]],
                spaces: vec![true, false],
                cats: BTreeMap::new(),
                flags: BTreeMap::new(),
                span_groups: vec![1, 2, 3, 4],
                user_data: Some(vec![5, 6]),
            }],
        }
    }

    #[test]
    fn rejects_compressed_and_decompressed_byte_limits() {
        let bytes = fixture().to_bytes().unwrap();
        for limits in [
            DocBinLimits {
                max_compressed_bytes: bytes.len() - 1,
                ..DocBinLimits::default()
            },
            DocBinLimits {
                max_decompressed_bytes: 1,
                ..DocBinLimits::default()
            },
        ] {
            assert!(matches!(
                DocBin::from_bytes_with_limits(&bytes, &limits),
                Err(DocBinError::LimitExceeded { .. })
            ));
        }
    }

    #[test]
    fn rejects_decoded_collection_limits() {
        let bytes = fixture().to_bytes().unwrap();
        for (limits, expected) in [
            (
                DocBinLimits {
                    max_documents: 0,
                    ..DocBinLimits::default()
                },
                DocBinLimitResource::Documents,
            ),
            (
                DocBinLimits {
                    max_tokens: 1,
                    ..DocBinLimits::default()
                },
                DocBinLimitResource::Tokens,
            ),
            (
                DocBinLimits {
                    max_attributes: 0,
                    ..DocBinLimits::default()
                },
                DocBinLimitResource::Attributes,
            ),
            (
                DocBinLimits {
                    max_strings: 0,
                    ..DocBinLimits::default()
                },
                DocBinLimitResource::Strings,
            ),
            (
                DocBinLimits {
                    max_metadata_bytes_per_document: 5,
                    ..DocBinLimits::default()
                },
                DocBinLimitResource::DocumentMetadataBytes,
            ),
        ] {
            assert!(matches!(
                DocBin::from_bytes_with_limits(&bytes, &limits),
                Err(DocBinError::LimitExceeded { resource, .. }) if resource == expected
            ));
        }
    }
}
