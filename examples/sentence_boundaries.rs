use jewel::{Bundle, NerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: sentence_boundaries <BUNDLE> <TEXT>")?;
    let text = args
        .next()
        .ok_or("usage: sentence_boundaries <BUNDLE> <TEXT>")?;
    if args.next().is_some() {
        return Err("usage: sentence_boundaries <BUNDLE> <TEXT>".into());
    }
    let text = text.to_str().ok_or("TEXT must be valid UTF-8")?;

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = NerPipeline::load(&bundle)?;
    let document = pipeline.process(text)?;

    println!("dependency_parser={}", pipeline.has_dependency_parser());
    println!("sentence_recognizer={}", pipeline.has_sentence_recognizer());
    println!("sentencizer={}", pipeline.has_sentencizer());
    for (index, token) in document.tokens().iter().enumerate() {
        println!(
            "{index}\tsent_start={}\t{}..{}\t{:?}",
            token.sent_start,
            token.idx,
            token.idx + token.text.chars().count(),
            token.text
        );
    }
    Ok(())
}
