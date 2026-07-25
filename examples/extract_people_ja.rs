use jewel_spacy::{Bundle, JapaneseNerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let path = args.next().ok_or("usage: extract_people_ja BUNDLE TEXT")?;
    let text = args
        .next()
        .ok_or("usage: extract_people_ja BUNDLE TEXT")?
        .into_string()
        .map_err(|_| "TEXT must be valid UTF-8")?;

    let bundle = Bundle::load(path)?;
    let pipeline = JapaneseNerPipeline::load(&bundle)?;
    for person in pipeline.extract_people(&text)? {
        println!(
            "{}\t{}..{}\t{}..{}",
            person.text, person.start_token, person.end_token, person.start_char, person.end_char
        );
    }
    Ok(())
}
