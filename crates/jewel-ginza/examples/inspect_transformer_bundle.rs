use jewel_core::{Bundle, DependencyParser, EntityRecognizer};
use jewel_ginza::{ginza_model_family, GinzaModelFamily, TransformerSpec};
use serde_json::json;

fn inspect(path: &std::path::Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let bundle = Bundle::load(path)?;
    if ginza_model_family(bundle.manifest())? != GinzaModelFamily::Electra {
        return Err("bundle is not GiNZA Electra".into());
    }
    let spec = TransformerSpec::from_bundle(&bundle)?;
    bundle.load_tokenizer()?;

    let parser_names = bundle
        .manifest()
        .pipeline
        .iter()
        .filter(|component| component.factory == "parser")
        .map(|component| component.name.as_str())
        .collect::<Vec<_>>();
    if parser_names.len() > 1 {
        return Err("bundle contains multiple parser components".into());
    }
    if let Some(parser) = parser_names.first() {
        DependencyParser::load(&bundle, parser)?;
    }
    let ner_names = bundle
        .manifest()
        .pipeline
        .iter()
        .filter(|component| component.factory == "ner")
        .map(|component| component.name.as_str())
        .collect::<Vec<_>>();
    let [ner] = ner_names.as_slice() else {
        return Err("bundle must contain exactly one NER component".into());
    };
    EntityRecognizer::load(&bundle, ner)?;

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
