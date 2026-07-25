fn code_point_to_byte(text: &str, offset: usize) -> Option<usize> {
    if offset == text.chars().count() {
        return Some(text.len());
    }
    text.char_indices().nth(offset).map(|(byte, _)| byte)
}

fn slice_code_points(text: &str, start: usize, end: usize) -> Option<&str> {
    let start_byte = code_point_to_byte(text, start)?;
    let end_byte = code_point_to_byte(text, end)?;
    text.get(start_byte..end_byte)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let text = args
        .next()
        .ok_or("usage: unicode_offsets <TEXT> <START_CHAR> <END_CHAR>")?;
    let start = args
        .next()
        .ok_or("usage: unicode_offsets <TEXT> <START_CHAR> <END_CHAR>")?
        .parse::<usize>()?;
    let end = args
        .next()
        .ok_or("usage: unicode_offsets <TEXT> <START_CHAR> <END_CHAR>")?
        .parse::<usize>()?;
    if args.next().is_some() || start > end {
        return Err("usage: unicode_offsets <TEXT> <START_CHAR> <END_CHAR>".into());
    }

    // Jewel follows spaCy and reports Unicode code-point offsets. Rust string
    // slicing uses UTF-8 byte offsets, so convert before indexing source text.
    let slice = slice_code_points(&text, start, end).ok_or("offset is outside the text")?;
    println!("{slice}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{code_point_to_byte, slice_code_points};

    #[test]
    fn converts_code_point_offsets_for_multibyte_text() {
        let text = "契約者は山田太郎です";
        assert_eq!(slice_code_points(text, 4, 8), Some("山田太郎"));
        assert_eq!(code_point_to_byte(text, 4), Some(12));
        assert_eq!(code_point_to_byte(text, 8), Some(24));
    }

    #[test]
    fn accepts_the_end_of_the_string() {
        let text = "Acme東京";
        assert_eq!(slice_code_points(text, 4, 6), Some("東京"));
        assert_eq!(code_point_to_byte(text, 6), Some(text.len()));
    }

    #[test]
    fn rejects_offsets_outside_the_string() {
        assert_eq!(slice_code_points("東京", 0, 3), None);
        assert_eq!(code_point_to_byte("東京", 3), None);
    }
}
