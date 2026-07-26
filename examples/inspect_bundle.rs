use jewel::{Bundle, NerCompatibilityReport};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let first = args
        .next()
        .ok_or("usage: inspect_bundle [--json] <BUNDLE>")?;
    let (json, bundle_path) = if first == "--json" {
        (
            true,
            args.next()
                .ok_or("usage: inspect_bundle [--json] <BUNDLE>")?,
        )
    } else {
        (false, first)
    };
    if args.next().is_some() {
        return Err("usage: inspect_bundle [--json] <BUNDLE>".into());
    }

    let report = NerCompatibilityReport::inspect(&bundle_path);
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        if !report.compatible {
            std::process::exit(1);
        }
        return Ok(());
    }
    if let Some(diagnostic) = report.diagnostics.first() {
        eprintln!(
            "incompatible: {} ({:?}): {}",
            diagnostic.code, diagnostic.area, diagnostic.message
        );
        std::process::exit(1);
    }

    // The compatibility inspection above constructs the language-aware NER
    // pipeline. Loading again keeps this example's human-readable summary
    // focused on the public Bundle API.
    let bundle = Bundle::load(bundle_path)?;
    let manifest = bundle.manifest();

    println!("bundle: {}", bundle.root().display());
    println!("NER compatible: yes");
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
