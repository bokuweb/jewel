use std::env;

use jewel_core::Bundle;
use jewel_ginza::GinzaPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let bundle_path = arguments
        .next()
        .ok_or("usage: extract_entities BUNDLE TEXT")?;
    let text = arguments
        .next()
        .ok_or("usage: extract_entities BUNDLE TEXT")?;
    if arguments.next().is_some() {
        return Err("usage: extract_entities BUNDLE TEXT".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = GinzaPipeline::load(&bundle)?;
    println!("ene_label\tcoarse_label\ttext\ttokens\tcharacters");
    for entity in pipeline.extract_entities(&text)? {
        let span = &entity.entity;
        println!(
            "{}\t{}\t{}\t{}..{}\t{}..{}",
            entity.ene_label(),
            entity.coarse_label.unwrap_or("-"),
            span.text,
            span.start_token,
            span.end_token,
            span.start_char,
            span.end_char,
        );
    }
    Ok(())
}
