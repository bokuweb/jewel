use base64::{engine::general_purpose::STANDARD, Engine as _};
use jewel_core::{attrs, hash_string, DocBin};

const DOCBIN_FIXTURE: &str = include_str!("fixtures/docbin_v3_8.b64");

#[test]
fn reads_spacy_3_8_docbin_fixture() {
    let bytes = STANDARD.decode(DOCBIN_FIXTURE.trim()).unwrap();
    let doc_bin = DocBin::from_bytes(&bytes).unwrap();

    assert_eq!(doc_bin.version(), "0.1");
    assert_eq!(doc_bin.attrs().first(), Some(&attrs::ORTH));
    assert_eq!(doc_bin.records().len(), 2);
    assert_eq!(doc_bin.records()[0].tokens.len(), 4);
    assert_eq!(doc_bin.records()[1].tokens.len(), 7);
    assert_eq!(doc_bin.records()[0].spaces, [false, true, false, false]);
    assert_eq!(doc_bin.records()[0].tokens[0][0], hash_string("Hello"));
    assert!(doc_bin.strings().iter().any(|text| text == "日本語"));
}

#[test]
fn rust_roundtrip_preserves_spacy_fixture_semantics() {
    let bytes = STANDARD.decode(DOCBIN_FIXTURE.trim()).unwrap();
    let original = DocBin::from_bytes(&bytes).unwrap();
    let rust_bytes = original.to_bytes().unwrap();
    let roundtrip = DocBin::from_bytes(&rust_bytes).unwrap();

    assert_eq!(roundtrip, original);
}
