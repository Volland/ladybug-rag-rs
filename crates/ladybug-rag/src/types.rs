use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A chunk of text from a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    pub id: String,
    pub text: String,
    pub source: String,
    pub position: usize,
    pub embedding: Vec<f32>,
}

/// An extracted entity (concept / term).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRecord {
    pub id: String,
    pub label: String,
    pub entity_type: EntityType,
    pub description: String,
    pub embedding: Vec<f32>,
    pub pagerank: f64,
    pub community: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityType {
    Concept,
    Term,
}

/// A relation between two entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub source_id: String,
    pub target_id: String,
    pub relation_type: String,
}

/// A retrieval result with its score and provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalResult {
    pub chunk: ChunkRecord,
    pub score: f64,
    pub method: RetrievalMethod,
    pub entities: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetrievalMethod {
    Vector,
    Graph,
    Hybrid,
}

/// Statistics about the RAG knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagStats {
    pub chunk_count: usize,
    pub entity_count: usize,
    pub mention_count: usize,
    pub relation_count: usize,
    pub community_count: u64,
}

/// Generate a deterministic ID from content.
pub fn make_id(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}
