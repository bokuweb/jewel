use jewel::Bundle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args.next().ok_or("usage: inspect_bundle <BUNDLE>")?;
    if args.next().is_some() {
        return Err("usage: inspect_bundle <BUNDLE>".into());
    }

    // Bundle::load performs structural validation, verifies referenced files
    // and tensors, and rejects bundles that declare a Python dependency.
    let bundle = Bundle::load(bundle_path)?;
    let manifest = bundle.manifest();

    println!("bundle: {}", bundle.root().display());
    println!("format version: {}", manifest.format_version);
    println!(
        "source: {} {} (spaCy {}, language {})",
        manifest.source.model_name,
        manifest.source.model_version,
        manifest.source.spacy_version,
        manifest.source.lang
    );
    println!(
        "runtime: minimum {}, requires Python: {}",
        manifest.runtime.min_runtime_version, manifest.runtime.requires_python
    );
    println!(
        "tokenizer: {:?} ({})",
        manifest.tokenizer.kind, manifest.tokenizer.path
    );
    println!("components:");

    for component in &manifest.pipeline {
        let tensor_count = component
            .nodes
            .iter()
            .map(|node| node.params.len())
            .sum::<usize>();
        println!(
            "  {}: factory={}, kind={:?}, nodes={}, tensors={}, labels={}, moves={}",
            component.name,
            component.factory,
            component.kind,
            component.nodes.len(),
            tensor_count,
            component.labels.len(),
            component.moves.len()
        );
    }

    Ok(())
}
