use std::hint::black_box;
use std::time::Instant;

use jewel::{Bundle, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: benchmark_tokenizer <BUNDLE> <ITERATIONS> <TEXT>")?;
    let iterations = args
        .next()
        .ok_or("usage: benchmark_tokenizer <BUNDLE> <ITERATIONS> <TEXT>")?
        .to_str()
        .ok_or("ITERATIONS must be valid UTF-8")?
        .parse::<usize>()?;
    let text = args
        .next()
        .ok_or("usage: benchmark_tokenizer <BUNDLE> <ITERATIONS> <TEXT>")?;
    if args.next().is_some() || iterations == 0 {
        return Err("usage: benchmark_tokenizer <BUNDLE> <ITERATIONS> <TEXT>".into());
    }
    let text = text.to_str().ok_or("TEXT must be valid UTF-8")?;

    let bundle = Bundle::load(bundle_path)?;
    let tokenizer = bundle.load_tokenizer()?;
    let mut session = Tokenizer::session(&tokenizer);
    for _ in 0..10 {
        black_box(session.tokenize(black_box(text))?);
    }

    let start = Instant::now();
    let mut token_count = 0_usize;
    for _ in 0..iterations {
        token_count += black_box(session.tokenize(black_box(text))?).len();
    }
    let elapsed = start.elapsed();
    let nanos_per_document = elapsed.as_nanos() as f64 / iterations as f64;
    println!(
        "iterations={iterations} tokens={} elapsed_ms={:.3} us_per_document={:.3}",
        token_count,
        elapsed.as_secs_f64() * 1_000.0,
        nanos_per_document / 1_000.0
    );
    Ok(())
}
