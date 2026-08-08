use jewel_core::Bundle;
use jewel_ginza::validate_electra_bundle;
use serde_json::json;

fn inspect(path: &std::path::Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let bundle = Bundle::load(path)?;
    let spec = validate_electra_bundle(&bundle)?;
    let entity_rulers = bundle
        .manifest()
        .pipeline
        .iter()
        .filter(|component| component.factory == "entity_ruler")
        .map(|component| component.name.clone())
        .collect::<Vec<_>>();
    let sentence_boundary =
        bundle
            .manifest()
            .pipeline
            .iter()
            .find(|component| component.factory == "parser")
            .or_else(|| {
                bundle.manifest().pipeline.iter().find(|component| {
                    matches!(component.factory.as_str(), "senter" | "sentencizer")
                })
            })
            .map(|component| {
                json!({
                    "name": component.name,
                    "factory": component.factory,
                })
            });

    Ok(json!({
        "report_version": 1,
        "compatible": true,
        "bundle_path": path,
        "source": bundle.manifest().source,
        "transformer": {
            "architecture": spec.architecture,
            "model": spec.model,
            "hidden_width": spec.hidden_width,
            "window": spec.window,
            "stride": spec.stride,
            "max_wordpieces": spec.max_wordpieces,
        },
        "sentence_boundary": sentence_boundary,
        "entity_rulers": entity_rulers,
        "diagnostics": [],
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let first = args
        .next()
        .ok_or("usage: inspect_transformer_bundle [--json] <BUNDLE>")?;
    let path = if first == "--json" {
        args.next()
            .ok_or("usage: inspect_transformer_bundle [--json] <BUNDLE>")?
    } else {
        first
    };
    if args.next().is_some() {
        return Err("usage: inspect_transformer_bundle [--json] <BUNDLE>".into());
    }
    let path = std::path::PathBuf::from(path);
    match inspect(&path) {
        Ok(report) => {
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Err(error) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "report_version": 1,
                    "compatible": false,
                    "bundle_path": path,
                    "diagnostics": [{
                        "code": "transformer_bundle_incompatible",
                        "area": "component",
                        "component": "transformer",
                        "message": error.to_string(),
                    }],
                }))?
            );
            std::process::exit(1);
        }
    }
}
