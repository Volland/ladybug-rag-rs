use regex::Regex;
use std::collections::HashSet;

use crate::types::{EntityType, Relation};

/// An extracted entity before embedding.
#[derive(Debug, Clone)]
pub struct RawEntity {
    pub label: String,
    pub entity_type: EntityType,
}

/// Extract entities from text using a simple regex-based approach.
///
/// Identifies capitalized noun phrases as CONCEPT or TERM entities.
/// This mirrors the Python `extract_entities_simple()` implementation.
pub fn extract_entities(text: &str) -> Vec<RawEntity> {
    let re = Regex::new(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)").unwrap();
    let stop_words: HashSet<&str> = [
        "The", "This", "That", "These", "Those", "When", "Where", "Which",
        "While", "With", "Would", "Could", "Should", "Have", "Has", "Had",
        "Does", "Did", "Will", "Shall", "May", "Might", "Must", "Can",
        "Being", "Been", "Are", "Were", "Was", "Not", "But", "And", "For",
        "From", "Into", "Then", "Than", "Also", "Each", "Every", "Both",
        "Few", "More", "Most", "Other", "Some", "Such", "Only", "Over",
        "After", "Before", "Between", "Under", "Again", "Further", "Once",
        "Here", "There", "All", "Any", "Many", "Much", "Our", "Its",
    ]
    .into_iter()
    .collect();

    let mut seen = HashSet::new();
    let mut entities = Vec::new();

    for cap in re.captures_iter(text) {
        let label = cap[1].to_string();

        if stop_words.contains(label.as_str()) {
            continue;
        }
        if label.len() < 3 {
            continue;
        }

        let key = label.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);

        let entity_type = if label.contains(' ') {
            EntityType::Concept
        } else {
            EntityType::Term
        };

        entities.push(RawEntity { label, entity_type });
    }

    entities
}

/// Extract co-occurrence relations between entities within the same sentence.
pub fn extract_relations(text: &str, entity_labels: &[String]) -> Vec<Relation> {
    let mut relations = Vec::new();
    let label_set: HashSet<&str> = entity_labels.iter().map(|s| s.as_str()).collect();

    for sentence in text.split(['.', '!', '?']) {
        let mut found_in_sentence = Vec::new();
        for label in &label_set {
            if sentence.contains(label) {
                found_in_sentence.push(label.to_string());
            }
        }

        // Create co-occurrence relations between all pairs
        for i in 0..found_in_sentence.len() {
            for j in (i + 1)..found_in_sentence.len() {
                let (a, b) = if found_in_sentence[i] < found_in_sentence[j] {
                    (&found_in_sentence[i], &found_in_sentence[j])
                } else {
                    (&found_in_sentence[j], &found_in_sentence[i])
                };
                relations.push(Relation {
                    source_id: crate::types::make_id(a),
                    target_id: crate::types::make_id(b),
                    relation_type: "co_occurs_with".to_string(),
                });
            }
        }
    }

    // Deduplicate
    let mut seen = HashSet::new();
    relations.retain(|r| {
        let key = format!("{}:{}:{}", r.source_id, r.target_id, r.relation_type);
        seen.insert(key)
    });

    relations
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_entities() {
        let text = "Machine Learning is a subset of Artificial Intelligence. \
                     Deep Learning uses Neural Networks.";
        let entities = extract_entities(text);
        let labels: Vec<&str> = entities.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains(&"Machine Learning"));
        assert!(labels.contains(&"Artificial Intelligence"));
        assert!(labels.contains(&"Deep Learning"));
        assert!(labels.contains(&"Neural Networks"));
    }

    #[test]
    fn test_extract_relations() {
        let text = "Machine Learning and Deep Learning are both used in AI.";
        let labels = vec![
            "Machine Learning".to_string(),
            "Deep Learning".to_string(),
        ];
        let relations = extract_relations(text, &labels);
        assert!(!relations.is_empty());
        assert_eq!(relations[0].relation_type, "co_occurs_with");
    }

    #[test]
    fn test_stop_words_filtered() {
        let text = "The quick brown fox. This is a test. When things happen.";
        let entities = extract_entities(text);
        let labels: Vec<&str> = entities.iter().map(|e| e.label.as_str()).collect();
        assert!(!labels.contains(&"The"));
        assert!(!labels.contains(&"This"));
        assert!(!labels.contains(&"When"));
    }
}
