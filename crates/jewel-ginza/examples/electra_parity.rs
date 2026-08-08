use jewel_core::Bundle;
use jewel_ginza::{CandleElectraEncoder, GinzaElectraPipeline};
use serde::Deserialize;

#[derive(Deserialize)]
struct Corpus {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    text: String,
    tokens: Vec<String>,
    entities: Vec<ExpectedEntity>,
}

#[derive(Debug, Deserialize, Eq, PartialEq)]
struct ExpectedEntity {
    text: String,
    label: String,
    start_token: usize,
    end_token: usize,
    start_char: usize,
    end_char: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: electra_parity <BUNDLE> <CORPUS>")?;
    let corpus_path = args
        .next()
        .ok_or("usage: electra_parity <BUNDLE> <CORPUS>")?;
    if args.next().is_some() {
        return Err("usage: electra_parity <BUNDLE> <CORPUS>".into());
    }

    let corpus: Corpus = serde_json::from_slice(&std::fs::read(corpus_path)?)?;
    let bundle = Bundle::load(bundle_path)?;
    let encoder = CandleElectraEncoder::load(&bundle)?;
    let pipeline = GinzaElectraPipeline::load(&bundle, encoder)?;
    let texts = corpus
        .cases
        .iter()
        .map(|case| case.text.as_str())
        .collect::<Vec<_>>();
    let docs = pipeline.process_batch(&texts)?;
    let mut mismatches = Vec::new();
    let mut entity_count = 0;
    for (index, (case, doc)) in corpus.cases.iter().zip(&docs).enumerate() {
        let tokens = doc
            .tokens()
            .iter()
            .map(|token| token.text.to_string())
            .collect::<Vec<_>>();
        let entities = pipeline
            .entities(doc)
            .into_iter()
            .map(|entity| ExpectedEntity {
                text: entity.entity.text,
                label: entity.entity.label,
                start_token: entity.entity.start_token,
                end_token: entity.entity.end_token,
                start_char: entity.entity.start_char,
                end_char: entity.entity.end_char,
            })
            .collect::<Vec<_>>();
        entity_count += case.entities.len();
        if tokens != case.tokens || entities != case.entities {
            mismatches.push(serde_json::json!({
                "case": index,
                "text": case.text,
                "expected_tokens": case.tokens,
                "actual_tokens": tokens,
                "expected_entities": format!("{:?}", case.entities),
                "actual_entities": format!("{entities:?}"),
            }));
        }
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "cases": corpus.cases.len(),
            "entities": entity_count,
            "mismatches": mismatches,
        }))?
    );
    if mismatches.is_empty() {
        Ok(())
    } else {
        std::process::exit(1);
    }
}
