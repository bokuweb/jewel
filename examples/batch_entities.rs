use jewel::{Bundle, NerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: batch_entities <BUNDLE> <TEXT> [TEXT ...]")?;
    let texts = args
        .map(|value| {
            value
                .into_string()
                .map_err(|_| "every TEXT must be valid UTF-8")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if texts.is_empty() {
        return Err("usage: batch_entities <BUNDLE> <TEXT> [TEXT ...]".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = NerPipeline::load(&bundle)?;
    let batches = pipeline.extract_entities_batch(&texts)?;

    println!("language: {}", pipeline.language().code());
    for (index, (text, entities)) in texts.iter().zip(batches).enumerate() {
        println!("document {index}: {text}");
        for entity in entities {
            println!(
                "  {}\t{}\t{}..{}",
                entity.label, entity.text, entity.start_char, entity.end_char
            );
        }
    }

    Ok(())
}
