use jewel::{Bundle, EnglishNerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: extract_entities_en <BUNDLE> <TEXT>")?;
    let text = args
        .next()
        .ok_or("usage: extract_entities_en <BUNDLE> <TEXT>")?
        .into_string()
        .map_err(|_| "TEXT must be valid UTF-8")?;
    if args.next().is_some() {
        return Err("usage: extract_entities_en <BUNDLE> <TEXT>".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = EnglishNerPipeline::load(&bundle)?;

    println!("label\ttext\ttokens\tcharacters");
    for entity in pipeline.extract_entities(&text)? {
        println!(
            "{}\t{}\t{}..{}\t{}..{}",
            entity.label,
            entity.text,
            entity.start_token,
            entity.end_token,
            entity.start_char,
            entity.end_char
        );
    }

    Ok(())
}
