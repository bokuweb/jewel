use std::{env, io};

use jewel_core::{Bundle, Tok2Vec};
use serde::Serialize;

#[derive(Serialize)]
struct Tok2VecOutput {
    tokens: Vec<String>,
    rows: usize,
    width: usize,
    vectors: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let bundle_path = arguments
        .next()
        .ok_or("usage: tok2vec_json BUNDLE COMPONENT")?;
    let component = arguments
        .next()
        .ok_or("usage: tok2vec_json BUNDLE COMPONENT")?;
    if arguments.next().is_some() {
        return Err("usage: tok2vec_json BUNDLE COMPONENT".into());
    }

    let bundle = Bundle::load(bundle_path)?;
    let tokenizer = bundle.load_tokenizer()?;
    let mut text = String::new();
    io::Read::read_to_string(&mut io::stdin().lock(), &mut text)?;
    let doc = tokenizer.tokenize(&text)?;
    let tok2vec = Tok2Vec::load(&bundle, &component)?;
    let vectors = tok2vec.forward(&doc)?;
    let output = Tok2VecOutput {
        tokens: doc
            .tokens()
            .iter()
            .map(|token| token.text.to_string())
            .collect(),
        rows: vectors.rows(),
        width: vectors.cols(),
        vectors: vectors.as_slice().to_vec(),
    };
    serde_json::to_writer(io::stdout().lock(), &output)?;
    Ok(())
}
