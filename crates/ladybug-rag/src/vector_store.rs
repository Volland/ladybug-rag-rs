use crate::embeddings::cosine_similarity;

/// A simple in-memory vector store for nearest-neighbor search.
///
/// Uses brute-force cosine similarity. For production, swap in
/// an HNSW implementation (e.g. `instant-distance` or `hnswlib`).
pub struct VectorStore {
    entries: Vec<VectorEntry>,
}

struct VectorEntry {
    id: String,
    embedding: Vec<f32>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Insert a vector with an associated ID.
    pub fn insert(&mut self, id: String, embedding: Vec<f32>) {
        // Update if exists
        if let Some(entry) = self.entries.iter_mut().find(|e| e.id == id) {
            entry.embedding = embedding;
        } else {
            self.entries.push(VectorEntry { id, embedding });
        }
    }

    /// Find the `top_k` most similar vectors to `query`.
    /// Returns (id, similarity_score) pairs, sorted by descending similarity.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .entries
            .iter()
            .map(|e| {
                let sim = cosine_similarity(query, &e.embedding);
                (e.id.clone(), sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut store = VectorStore::new();
        store.insert("a".into(), vec![1.0, 0.0, 0.0]);
        store.insert("b".into(), vec![0.0, 1.0, 0.0]);
        store.insert("c".into(), vec![0.9, 0.1, 0.0]);

        let results = store.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
        assert_eq!(results[1].0, "c");
    }
}
