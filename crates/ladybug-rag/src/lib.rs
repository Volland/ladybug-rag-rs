//! Hybrid Graph RAG — a Rust implementation of the ladybug-rag pattern.
//!
//! Combines four retrieval mechanisms:
//! 1. Vector search (HNSW-like cosine similarity)
//! 2. Graph traversal (entity relationships)
//! 3. PageRank scoring (entity importance)
//! 4. Louvain community detection (topic clustering)
//!
//! Results are fused using Reciprocal Rank Fusion (RRF).

pub mod chunker;
pub mod embeddings;
pub mod entities;
pub mod graph_store;
#[cfg(feature = "onnx")]
pub mod onnx_embedder;
pub mod rag;
pub mod types;
pub mod vector_store;

pub use rag::HybridGraphRag;
pub use types::*;
