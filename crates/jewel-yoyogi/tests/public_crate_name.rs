use jewel_core::Bundle;
use jewel_yoyogi::{coarse_label, GinzaError, GinzaPipeline};

#[test]
fn yoyogi_exposes_the_existing_ginza_adapter_api() {
    assert_eq!(env!("CARGO_PKG_NAME"), "jewel-yoyogi");
    let _load: fn(&Bundle) -> Result<GinzaPipeline, GinzaError> = GinzaPipeline::load;
    assert_eq!(coarse_label("Person"), Some("PERSON"));
    assert_eq!(coarse_label("Period_Time"), Some("TIME"));
}
