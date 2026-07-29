use std::env;

use jewel_core::{Bundle, EntityConstraint};
use jewel_ginza::GinzaPipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let bundle_path = arguments
        .next()
        .ok_or("usage: extract_entities BUNDLE TEXT [START:END:LABEL ...]")?;
    let text = arguments
        .next()
        .ok_or("usage: extract_entities BUNDLE TEXT [START:END:LABEL ...]")?;
    let constraints = arguments
        .map(|argument| parse_constraint(&argument))
        .collect::<Result<Vec<_>, _>>()?;

    let bundle = Bundle::load(bundle_path)?;
    let pipeline = GinzaPipeline::load(&bundle)?;
    println!("ene_label\tcoarse_label\ttext\ttokens\tcharacters");
    for entity in pipeline.extract_entities_with_constraints(&text, &constraints)? {
        let span = &entity.entity;
        println!(
            "{}\t{}\t{}\t{}..{}\t{}..{}",
            entity.ene_label(),
            entity.coarse_label.unwrap_or("-"),
            span.text,
            span.start_token,
            span.end_token,
            span.start_char,
            span.end_char,
        );
    }
    Ok(())
}

fn parse_constraint(value: &str) -> Result<EntityConstraint, Box<dyn std::error::Error>> {
    let mut fields = value.splitn(3, ':');
    let start = fields
        .next()
        .ok_or("constraint must be START:END:LABEL")?
        .parse()?;
    let end = fields
        .next()
        .ok_or("constraint must be START:END:LABEL")?
        .parse()?;
    let label = fields.next().ok_or("constraint must be START:END:LABEL")?;
    Ok(match label {
        "-" => EntityConstraint::Blocked { start, end },
        "O" => EntityConstraint::Outside { start, end },
        label => EntityConstraint::Entity {
            start,
            end,
            label: label.to_owned(),
        },
    })
}
