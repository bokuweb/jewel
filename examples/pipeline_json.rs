use std::{env, io};

use jewel_core::{Bundle, NerPipeline};
use serde::Serialize;

#[derive(Serialize)]
struct TokenOutput {
    text: String,
    idx: usize,
    head: i32,
    dep: u64,
    sent_start: i8,
    ent_iob: u8,
    ent_type: u64,
}

#[derive(Serialize)]
struct PipelineOutput {
    tokens: Vec<TokenOutput>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let bundle_path = arguments.next().ok_or("usage: pipeline_json BUNDLE")?;
    if arguments.next().is_some() {
        return Err("usage: pipeline_json BUNDLE".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = NerPipeline::load(&bundle)?;
    let mut text = String::new();
    io::Read::read_to_string(&mut io::stdin().lock(), &mut text)?;
    let doc = pipeline.process(&text)?;
    let output = PipelineOutput {
        tokens: doc
            .tokens()
            .iter()
            .map(|token| TokenOutput {
                text: token.text.to_string(),
                idx: token.idx,
                head: token.head,
                dep: token.dep,
                sent_start: token.sent_start,
                ent_iob: token.ent_iob,
                ent_type: token.ent_type,
            })
            .collect(),
    };
    serde_json::to_writer(io::stdout().lock(), &output)?;
    Ok(())
}
