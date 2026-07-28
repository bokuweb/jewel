use std::hint::black_box;
use std::time::Instant;

use jewel_core::{Bundle, NerPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: benchmark_ner <BUNDLE> <ITERATIONS> <TEXT> [BATCH_SIZE]";
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args.next().ok_or(USAGE)?;
    let iterations = args
        .next()
        .ok_or(USAGE)?
        .to_str()
        .ok_or("ITERATIONS must be valid UTF-8")?
        .parse::<usize>()?;
    let text = args.next().ok_or(USAGE)?;
    let batch_size = match args.next() {
        Some(value) => value
            .to_str()
            .ok_or("BATCH_SIZE must be valid UTF-8")?
            .parse::<usize>()?,
        None => 1,
    };
    if args.next().is_some() || iterations == 0 || batch_size == 0 {
        return Err(USAGE.into());
    }
    let text = text.to_str().ok_or("TEXT must be valid UTF-8")?;

    let bundle_start = Instant::now();
    let bundle = Bundle::load(bundle_path)?;
    let bundle_elapsed = bundle_start.elapsed();

    let pipeline_start = Instant::now();
    let pipeline = NerPipeline::load(&bundle)?;
    let pipeline_elapsed = pipeline_start.elapsed();

    let batch = vec![text; batch_size];
    for _ in 0..10 {
        black_box(pipeline.extract_entities_batch(black_box(&batch))?);
    }

    let inference_start = Instant::now();
    let mut entity_count = 0_usize;
    for _ in 0..iterations {
        entity_count += black_box(pipeline.extract_entities_batch(black_box(&batch))?)
            .iter()
            .map(Vec::len)
            .sum::<usize>();
    }
    let inference_elapsed = inference_start.elapsed();
    let document_count = iterations
        .checked_mul(batch_size)
        .ok_or("ITERATIONS * BATCH_SIZE overflowed")?;
    let nanos_per_document = inference_elapsed.as_nanos() as f64 / document_count as f64;

    println!(
        "bundle_load_ms={:.3} pipeline_load_ms={:.3} iterations={iterations} \
         batch_size={batch_size} documents={document_count} entities={entity_count} \
         inference_ms={:.3} us_per_document={:.3}",
        bundle_elapsed.as_secs_f64() * 1_000.0,
        pipeline_elapsed.as_secs_f64() * 1_000.0,
        inference_elapsed.as_secs_f64() * 1_000.0,
        nanos_per_document / 1_000.0
    );
    Ok(())
}
