use std::hint::black_box;
use std::time::Instant;

use jewel::{Bundle, NerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args
        .next()
        .ok_or("usage: benchmark_ner <BUNDLE> <ITERATIONS> <TEXT>")?;
    let iterations = args
        .next()
        .ok_or("usage: benchmark_ner <BUNDLE> <ITERATIONS> <TEXT>")?
        .to_str()
        .ok_or("ITERATIONS must be valid UTF-8")?
        .parse::<usize>()?;
    let text = args
        .next()
        .ok_or("usage: benchmark_ner <BUNDLE> <ITERATIONS> <TEXT>")?;
    if args.next().is_some() || iterations == 0 {
        return Err("usage: benchmark_ner <BUNDLE> <ITERATIONS> <TEXT>".into());
    }
    let text = text.to_str().ok_or("TEXT must be valid UTF-8")?;

    let bundle_start = Instant::now();
    let bundle = Bundle::load(bundle_path)?;
    let bundle_elapsed = bundle_start.elapsed();

    let pipeline_start = Instant::now();
    let pipeline = NerPipeline::load(&bundle)?;
    let pipeline_elapsed = pipeline_start.elapsed();

    for _ in 0..10 {
        black_box(pipeline.extract_entities(black_box(text))?);
    }

    let inference_start = Instant::now();
    let mut entity_count = 0_usize;
    for _ in 0..iterations {
        entity_count += black_box(pipeline.extract_entities(black_box(text))?).len();
    }
    let inference_elapsed = inference_start.elapsed();
    let nanos_per_document = inference_elapsed.as_nanos() as f64 / iterations as f64;

    println!(
        "bundle_load_ms={:.3} pipeline_load_ms={:.3} iterations={iterations} \
         entities={entity_count} inference_ms={:.3} us_per_document={:.3}",
        bundle_elapsed.as_secs_f64() * 1_000.0,
        pipeline_elapsed.as_secs_f64() * 1_000.0,
        inference_elapsed.as_secs_f64() * 1_000.0,
        nanos_per_document / 1_000.0
    );
    Ok(())
}
