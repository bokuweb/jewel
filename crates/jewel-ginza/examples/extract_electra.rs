use jewel_core::Bundle;
use jewel_ginza::{CandleElectraEncoder, GinzaElectraPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: extract_electra <BUNDLE> <TEXT>")?;
    let text = args
        .next()
        .ok_or("usage: extract_electra <BUNDLE> <TEXT> [SPAN_BATCH_SIZE]")?
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
        return Err("usage: extract_electra <BUNDLE> <TEXT> [SPAN_BATCH_SIZE]".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let encoder = match span_batch_size {
        Some(size) => CandleElectraEncoder::load_with_span_batch_size(&bundle, size)?,
        None => CandleElectraEncoder::load(&bundle)?,
    };
    let pipeline = GinzaElectraPipeline::load(&bundle, encoder)?;
    let entities = pipeline
        .extract_entities(&text)?
        .into_iter()
        .map(|entity| {
            serde_json::json!({
                "text": entity.entity.text,
                "label": entity.entity.label,
                "coarse_label": entity.coarse_label,
                "start_token": entity.entity.start_token,
                "end_token": entity.entity.end_token,
                "start_char": entity.entity.start_char,
                "end_char": entity.entity.end_char,
            })
        })
        .collect::<Vec<_>>();
    println!("{}", serde_json::to_string_pretty(&entities)?);
    Ok(())
}
