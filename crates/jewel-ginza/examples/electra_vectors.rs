use jewel_core::Bundle;
use jewel_ginza::{CandleElectraEncoder, TransformerEncoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: electra_vectors <BUNDLE> <TEXT>")?;
    let text = args
        .next()
        .ok_or("usage: electra_vectors <BUNDLE> <TEXT> [SPAN_BATCH_SIZE]")?
        .into_string()
        .map_err(|_| "TEXT must be valid UTF-8")?;
    let span_batch_size = args
        .next()
        .map(|value| {
            value
                .into_string()
                .map_err(|_| "SPAN_BATCH_SIZE must be valid UTF-8")?
                .parse::<usize>()
                .map_err(|_| "SPAN_BATCH_SIZE must be a positive integer")
        })
        .transpose()?;
    if args.next().is_some() {
        return Err("usage: electra_vectors <BUNDLE> <TEXT> [SPAN_BATCH_SIZE]".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let tokenizer = bundle.load_tokenizer()?;
    let doc = tokenizer.tokenize(&text)?;
    let encoder = match span_batch_size {
        Some(size) => CandleElectraEncoder::load_with_span_batch_size(&bundle, size)?,
        None => CandleElectraEncoder::load(&bundle)?,
    };
    let vectors = encoder.encode(&doc)?;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "span_batch_size": encoder.span_batch_size(),
            "tokens": doc.tokens().iter().map(|token| token.text.as_ref()).collect::<Vec<_>>(),
            "wordpiece_ids": encoder.token_wordpiece_ids(&doc)?,
            "vectors": (0..vectors.rows()).map(|row| vectors.row(row)).collect::<Vec<_>>(),
        }))?
    );
    Ok(())
}
