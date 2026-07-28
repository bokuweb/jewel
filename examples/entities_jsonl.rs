use std::io::{self, BufRead, Write};

use jewel_core::{Bundle, NamedEntity, NerLanguage, NerPipeline};
use serde::Serialize;

#[derive(Serialize)]
struct DocumentOutput<'a> {
    text: &'a str,
    language: NerLanguage,
    entities: Vec<NamedEntity>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: entities_jsonl <BUNDLE> [--json-input]")?;
    let json_input = match args.next() {
        None => false,
        Some(value) if value == "--json-input" => true,
        Some(_) => return Err("usage: entities_jsonl <BUNDLE> [--json-input]".into()),
    };
    if args.next().is_some() {
        return Err("usage: entities_jsonl <BUNDLE> [--json-input]".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = NerPipeline::load(&bundle)?;
    let language = pipeline.language();
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    // Each non-empty input line is one document. The bundle and pipeline are
    // loaded once, which mirrors a long-running service or worker process.
    for line in stdin.lock().lines() {
        let input = line?;
        if input.trim().is_empty() {
            continue;
        }
        let text = if json_input {
            serde_json::from_str::<String>(&input)?
        } else {
            input
        };
        let entities = pipeline.extract_entities(&text)?;
        let output = DocumentOutput {
            text: &text,
            language,
            entities,
        };
        serde_json::to_writer(&mut stdout, &output)?;
        writeln!(stdout)?;
    }

    Ok(())
}
