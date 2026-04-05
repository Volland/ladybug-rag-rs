/// Trait for embedding providers.
///
/// Implementations can use local models (e.g. via ONNX runtime),
/// remote APIs (OpenAI, etc.), or simple heuristics for testing.
pub trait Embedder: Send + Sync {
    /// Embed a single text string into a vector.
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Embed multiple texts. Default implementation calls `embed` in a loop.
    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// The dimensionality of produced embeddings.
    fn dimension(&self) -> usize;
}

/// A simple bag-of-words embedder for testing and prototyping.
///
/// Creates sparse-ish vectors based on character trigram hashing.
/// Not suitable for production — use a real model.
pub struct SimpleEmbedder {
    dim: usize,
}

impl SimpleEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Default for SimpleEmbedder {
    fn default() -> Self {
        Self::new(384)
    }
}

impl Embedder for SimpleEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0.0f32; self.dim];
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();

        // Character trigram hashing
        for window in chars.windows(3) {
            let hash = window.iter().fold(0u64, |acc, &c| {
                acc.wrapping_mul(31).wrapping_add(c as u64)
            });
            let idx = (hash as usize) % self.dim;
            vec[idx] += 1.0;
        }

        // Word unigram hashing
        for word in lower.split_whitespace() {
            let hash = word.bytes().fold(0u64, |acc, b| {
                acc.wrapping_mul(37).wrapping_add(b as u64)
            });
            let idx = (hash as usize) % self.dim;
            vec[idx] += 2.0;
        }

        // L2 normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vec {
                *v /= norm;
            }
        }

        vec
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_embedder() {
        let emb = SimpleEmbedder::default();
        let v = emb.embed("hello world");
        assert_eq!(v.len(), 384);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_self_similarity() {
        let emb = SimpleEmbedder::default();
        let v = emb.embed("test text");
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_similar_texts_higher_similarity() {
        let emb = SimpleEmbedder::default();
        let a = emb.embed("machine learning algorithms");
        let b = emb.embed("machine learning models");
        let c = emb.embed("cooking recipe for pasta");
        let sim_ab = cosine_similarity(&a, &b);
        let sim_ac = cosine_similarity(&a, &c);
        assert!(sim_ab > sim_ac);
    }
}
