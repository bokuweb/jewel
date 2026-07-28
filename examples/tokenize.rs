use jewel_core::Bundle;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let bundle_path = args.next().ok_or("usage: tokenize <BUNDLE> <TEXT>")?;
    let text = args.next().ok_or("usage: tokenize <BUNDLE> <TEXT>")?;
    if args.next().is_some() {
        return Err("usage: tokenize <BUNDLE> <TEXT>".into());
    }
    let text = text.to_str().ok_or("TEXT must be valid UTF-8")?;

    let bundle = Bundle::load(bundle_path)?;
    let tokenizer = bundle.load_tokenizer()?;
    let document = tokenizer.tokenize(text)?;

    for (index, token) in document.tokens().iter().enumerate() {
        println!(
            "{index}\t{}..{}\t{:?}\tspace={}",
            token.idx,
            token.idx + token.text.chars().count(),
            token.text,
            token.has_space
        );
    }
    Ok(())
}
