use std::io::{self, BufRead, Write};

use jewel_spacy::{Bundle, EnglishNerPipeline, JapaneseNerPipeline, NamedEntity};
use serde::Serialize;

enum Pipeline {
    English(EnglishNerPipeline),
    Japanese(JapaneseNerPipeline),
}

impl Pipeline {
    fn load(bundle: &Bundle) -> Result<Self, Box<dyn std::error::Error>> {
        match bundle.manifest().source.lang.as_str() {
            "en" => Ok(Self::English(EnglishNerPipeline::load(bundle)?)),
            "ja" => Ok(Self::Japanese(JapaneseNerPipeline::load(bundle)?)),
            language => {
                Err(format!("unsupported bundle language {language:?}; expected en or ja").into())
            }
        }
    }

    fn extract(&self, text: &str) -> Result<Vec<NamedEntity>, Box<dyn std::error::Error>> {
        match self {
            Self::English(pipeline) => Ok(pipeline.extract_entities(text)?),
            Self::Japanese(pipeline) => Ok(pipeline.extract_entities(text)?),
        }
    }
}

#[derive(Serialize)]
struct EntityOutput {
    text: String,
    label: String,
    start_token: usize,
    end_token: usize,
    start_char: usize,
    end_char: usize,
}

impl From<NamedEntity> for EntityOutput {
    fn from(entity: NamedEntity) -> Self {
        Self {
            text: entity.text,
            label: entity.label,
            start_token: entity.start_token,
            end_token: entity.end_token,
            start_char: entity.start_char,
            end_char: entity.end_char,
        }
    }
}

#[derive(Serialize)]
struct DocumentOutput<'a> {
    text: &'a str,
    language: &'a str,
    entities: Vec<EntityOutput>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args.next().ok_or("usage: entities_jsonl <BUNDLE>")?;
    if args.next().is_some() {
        return Err("usage: entities_jsonl <BUNDLE>".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = Pipeline::load(&bundle)?;
    let language = bundle.manifest().source.lang.as_str();
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    // Each non-empty input line is one document. The bundle and pipeline are
    // loaded once, which mirrors a long-running service or worker process.
    for line in stdin.lock().lines() {
        let text = line?;
        if text.trim().is_empty() {
            continue;
        }
        let entities = pipeline
            .extract(&text)?
            .into_iter()
            .map(EntityOutput::from)
            .collect();
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
