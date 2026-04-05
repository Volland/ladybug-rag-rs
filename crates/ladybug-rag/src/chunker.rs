/// Paragraph-aware text chunker with overlap.
///
/// Splits text into chunks of approximately `chunk_size` characters,
/// respecting paragraph boundaries where feasible, with `overlap` characters
/// of context carried between chunks.
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut current = String::new();

    for para in &paragraphs {
        let trimmed = para.trim();
        if trimmed.is_empty() {
            continue;
        }

        if !current.is_empty() && current.len() + trimmed.len() + 2 > chunk_size {
            chunks.push(current.clone());

            // Carry overlap from end of current chunk
            let overlap_start = current.len().saturating_sub(overlap);
            current = current[overlap_start..].to_string();
            current.push_str("\n\n");
        }

        if !current.is_empty() && !current.ends_with("\n\n") {
            current.push_str("\n\n");
        }
        current.push_str(trimmed);
    }

    if !current.trim().is_empty() {
        chunks.push(current);
    }

    // If no paragraph splits happened, split by character boundary
    if chunks.is_empty() && !text.trim().is_empty() {
        let text = text.trim();
        let mut start = 0;
        while start < text.len() {
            let end = (start + chunk_size).min(text.len());
            // Try to break at a word boundary
            let actual_end = if end < text.len() {
                text[start..end]
                    .rfind(char::is_whitespace)
                    .map(|p| start + p + 1)
                    .unwrap_or(end)
            } else {
                end
            };
            chunks.push(text[start..actual_end].to_string());
            start = actual_end.saturating_sub(overlap);
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_text(text, 40, 10);
        assert!(!chunks.is_empty());
        // All text should be represented
        for chunk in &chunks {
            assert!(!chunk.trim().is_empty());
        }
    }

    #[test]
    fn test_empty() {
        assert!(chunk_text("", 100, 10).is_empty());
    }

    #[test]
    fn test_single_paragraph() {
        let chunks = chunk_text("Hello world", 100, 10);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }
}
