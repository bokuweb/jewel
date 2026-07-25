use std::env;

use jewel::{Bundle, NerPipeline};

const SIGNATURE_LABELS: &[&str] = &["PERSON", "ORG", "GPE", "LOC", "FAC", "TITLE_AFFIX"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: extract_signature_entities <bundle> <text>")?;
    let text = args
        .next()
        .ok_or("usage: extract_signature_entities <bundle> <text>")?;

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = NerPipeline::load(&bundle)?;
    for entity in pipeline.extract_entities_by_labels(&text, SIGNATURE_LABELS)? {
        println!(
            "{}\t{}\t{}..{}",
            entity.label, entity.text, entity.start_char, entity.end_char
        );
    }
    Ok(())
}
