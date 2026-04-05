use std::collections::HashMap;

use crate::chunker::chunk_text;
use crate::embeddings::Embedder;
use crate::entities::{extract_entities, extract_relations};
use crate::graph_store::GraphStore;
use crate::types::*;
use crate::vector_store::VectorStore;

/// Configuration for the Hybrid Graph RAG engine.
pub struct RagConfig {
    /// Characters per chunk.
    pub chunk_size: usize,
    /// Overlap between chunks.
    pub chunk_overlap: usize,
    /// PageRank damping factor.
    pub pagerank_damping: f64,
    /// PageRank convergence tolerance.
    pub pagerank_tolerance: f64,
    /// Louvain resolution parameter.
    pub louvain_gamma: f64,
    /// Number of hops for graph expansion.
    pub graph_hops: usize,
    /// RRF k parameter.
    pub rrf_k: f64,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            chunk_size: 2048,
            chunk_overlap: 200,
            pagerank_damping: 0.85,
            pagerank_tolerance: 1e-6,
            louvain_gamma: 1.0,
            graph_hops: 2,
            rrf_k: 60.0,
        }
    }
}

/// Hybrid Graph RAG engine.
///
/// Combines vector search, graph traversal, PageRank scoring,
/// and Louvain community detection with Reciprocal Rank Fusion.
pub struct HybridGraphRag {
    config: RagConfig,
    embedder: Box<dyn Embedder>,
    chunk_store: VectorStore,
    entity_store: VectorStore,
    graph: GraphStore,
    chunks: HashMap<String, ChunkRecord>,
    entities: HashMap<String, EntityRecord>,
    pagerank_scores: HashMap<String, f64>,
    community_assignments: HashMap<String, u64>,
    community_count: u64,
}

impl HybridGraphRag {
    /// Create a new RAG engine with the given embedder and config.
    pub fn new(embedder: Box<dyn Embedder>, config: RagConfig) -> Self {
        Self {
            config,
            embedder,
            chunk_store: VectorStore::new(),
            entity_store: VectorStore::new(),
            graph: GraphStore::new(),
            chunks: HashMap::new(),
            entities: HashMap::new(),
            pagerank_scores: HashMap::new(),
            community_assignments: HashMap::new(),
            community_count: 0,
        }
    }

    /// Ingest a text document into the knowledge base.
    pub fn ingest_text(&mut self, text: &str, source: &str) {
        let text_chunks = chunk_text(text, self.config.chunk_size, self.config.chunk_overlap);

        let mut prev_chunk_id: Option<String> = None;

        for (pos, chunk_text) in text_chunks.iter().enumerate() {
            let chunk_id = make_id(&format!("{}:{}:{}", source, pos, chunk_text));
            let embedding = self.embedder.embed(chunk_text);

            let chunk = ChunkRecord {
                id: chunk_id.clone(),
                text: chunk_text.clone(),
                source: source.to_string(),
                position: pos,
                embedding: embedding.clone(),
            };

            // Store chunk
            self.chunk_store.insert(chunk_id.clone(), embedding);
            self.graph.add_chunk(&chunk_id);
            self.chunks.insert(chunk_id.clone(), chunk);

            // NEXT_CHUNK edge
            if let Some(prev_id) = &prev_chunk_id {
                self.graph.add_next_chunk(prev_id, &chunk_id);
            }
            prev_chunk_id = Some(chunk_id.clone());

            // Extract and store entities
            let raw_entities = extract_entities(chunk_text);
            let entity_labels: Vec<String> =
                raw_entities.iter().map(|e| e.label.clone()).collect();

            for raw in &raw_entities {
                let entity_id = make_id(&raw.label);
                let entity_embedding = self.embedder.embed(&raw.label);

                if !self.entities.contains_key(&entity_id) {
                    let entity = EntityRecord {
                        id: entity_id.clone(),
                        label: raw.label.clone(),
                        entity_type: raw.entity_type,
                        description: String::new(),
                        embedding: entity_embedding.clone(),
                        pagerank: 0.0,
                        community: 0,
                    };
                    self.entity_store
                        .insert(entity_id.clone(), entity_embedding);
                    self.graph.add_entity(&entity_id);
                    self.entities.insert(entity_id.clone(), entity);
                }

                // MENTIONS edge
                self.graph.add_mention(&chunk_id, &entity_id);
            }

            // Extract relations
            let relations = extract_relations(chunk_text, &entity_labels);
            for rel in &relations {
                self.graph.add_relation(&rel.source_id, &rel.target_id, 1.0);
            }
        }
    }

    /// Ingest multiple documents.
    pub fn ingest_documents(&mut self, documents: &[(&str, &str)]) {
        for (text, source) in documents {
            self.ingest_text(text, source);
        }
    }

    /// Compute graph scores (PageRank + Louvain communities).
    /// Call this after ingesting all documents.
    pub fn compute_graph_scores(&mut self) {
        // PageRank
        self.pagerank_scores = self
            .graph
            .compute_pagerank(self.config.pagerank_damping, self.config.pagerank_tolerance);

        // Update entity records with PageRank scores
        for (id, entity) in &mut self.entities {
            if let Some(&score) = self.pagerank_scores.get(id) {
                entity.pagerank = score;
            }
        }

        // Louvain communities
        let (assignments, count) = self.graph.compute_communities(self.config.louvain_gamma);
        self.community_count = count;
        self.community_assignments = assignments;

        // Update entity records with community assignments
        for (id, entity) in &mut self.entities {
            if let Some(&community) = self.community_assignments.get(id) {
                entity.community = community;
            }
        }
    }

    /// Query the knowledge base.
    ///
    /// Performs:
    /// 1. Vector search on chunk embeddings
    /// 2. Entity seed identification via entity vector search
    /// 3. Graph expansion from seed entities
    /// 4. Reciprocal Rank Fusion of results
    pub fn query(&self, query_text: &str, top_k: usize) -> Vec<RetrievalResult> {
        let query_embedding = self.embedder.embed(query_text);

        // Stage 1: Vector search on chunks
        let vector_results = self.chunk_store.search(&query_embedding, top_k * 2);

        // Stage 2: Find seed entities via entity vector search
        let entity_results = self.entity_store.search(&query_embedding, 5);
        let seed_entity_ids: Vec<String> = entity_results.iter().map(|(id, _)| id.clone()).collect();

        // Stage 3: Graph expansion from seeds
        let graph_results =
            self.graph
                .graph_expand(&seed_entity_ids, self.config.graph_hops, &self.pagerank_scores);

        // Stage 4: RRF fusion
        let fused = rrf_fusion(&vector_results, &graph_results, self.config.rrf_k);

        // Build retrieval results
        let mut results = Vec::new();
        for (chunk_id, score) in fused.into_iter().take(top_k) {
            if let Some(chunk) = self.chunks.get(&chunk_id) {
                let entities = self.graph.chunk_entities(&chunk_id);
                let entity_labels: Vec<String> = entities
                    .iter()
                    .filter_map(|eid| self.entities.get(eid).map(|e| e.label.clone()))
                    .collect();

                let method = if vector_results.iter().any(|(id, _)| id == &chunk_id)
                    && graph_results.iter().any(|(id, _)| id == &chunk_id)
                {
                    RetrievalMethod::Hybrid
                } else if graph_results.iter().any(|(id, _)| id == &chunk_id) {
                    RetrievalMethod::Graph
                } else {
                    RetrievalMethod::Vector
                };

                results.push(RetrievalResult {
                    chunk: chunk.clone(),
                    score,
                    method,
                    entities: entity_labels,
                });
            }
        }

        results
    }

    /// Build an LLM-ready context string from retrieval results.
    pub fn build_context(
        &self,
        results: &[RetrievalResult],
        include_neighbors: bool,
        max_chars: usize,
    ) -> String {
        let mut context = String::new();
        let mut total_chars = 0;

        for (i, result) in results.iter().enumerate() {
            let header = format!(
                "--- Source: {} (chunk {}, score: {:.4}, method: {:?}) ---\n",
                result.chunk.source, result.chunk.position, result.score, result.method
            );

            if total_chars + header.len() + result.chunk.text.len() > max_chars {
                break;
            }

            context.push_str(&header);
            context.push_str(&result.chunk.text);
            context.push('\n');
            total_chars += header.len() + result.chunk.text.len() + 1;

            if !result.entities.is_empty() {
                let entities_str = format!("Entities: {}\n", result.entities.join(", "));
                context.push_str(&entities_str);
                total_chars += entities_str.len();
            }

            if include_neighbors {
                let neighbors = self.graph.neighboring_chunks(&result.chunk.id);
                for nid in neighbors.iter().take(1) {
                    if let Some(neighbor_chunk) = self.chunks.get(nid) {
                        let prefix = format!("[neighbor chunk {}]: ", neighbor_chunk.position);
                        let available = max_chars.saturating_sub(total_chars + prefix.len());
                        if available > 50 {
                            let snippet_len = available.min(200);
                            let snippet: String =
                                neighbor_chunk.text.chars().take(snippet_len).collect();
                            context.push_str(&prefix);
                            context.push_str(&snippet);
                            context.push_str("...\n");
                            total_chars += prefix.len() + snippet_len + 4;
                        }
                    }
                }
            }

            if i < results.len() - 1 {
                context.push('\n');
                total_chars += 1;
            }
        }

        context
    }

    /// Get community summary context for map-reduce style queries.
    pub fn get_community_summary(&self) -> HashMap<u64, Vec<String>> {
        let mut communities: HashMap<u64, Vec<String>> = HashMap::new();
        for entity in self.entities.values() {
            communities
                .entry(entity.community)
                .or_default()
                .push(entity.label.clone());
        }
        communities
    }

    /// Get statistics about the knowledge base.
    pub fn stats(&self) -> RagStats {
        RagStats {
            chunk_count: self.chunks.len(),
            entity_count: self.graph.entity_count(),
            mention_count: self.graph.mention_count(),
            relation_count: self.graph.relation_count(),
            community_count: self.community_count,
        }
    }
}

/// Reciprocal Rank Fusion: merge two ranked lists.
///
/// `score = sum(1 / (rank + k))` for each list where the item appears.
fn rrf_fusion(
    vector_ranked: &[(String, f32)],
    graph_ranked: &[(String, f64)],
    k: f64,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    for (rank, (id, _)) in vector_ranked.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (rank as f64 + k);
    }

    for (rank, (id, _)) in graph_ranked.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (rank as f64 + k);
    }

    let mut fused: Vec<(String, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::SimpleEmbedder;

    fn make_rag() -> HybridGraphRag {
        HybridGraphRag::new(
            Box::new(SimpleEmbedder::default()),
            RagConfig::default(),
        )
    }

    #[test]
    fn test_ingest_and_query() {
        let mut rag = make_rag();

        rag.ingest_text(
            "Machine Learning is a branch of Artificial Intelligence. \
             Deep Learning is a subset of Machine Learning that uses Neural Networks. \
             Natural Language Processing enables computers to understand human language.",
            "test.md",
        );

        rag.compute_graph_scores();

        let stats = rag.stats();
        assert!(stats.chunk_count > 0);
        assert!(stats.entity_count > 0);

        let results = rag.query("What is Machine Learning?", 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_build_context() {
        let mut rag = make_rag();
        rag.ingest_text(
            "Rust is a systems programming language. \
             Rust provides memory safety without garbage collection.",
            "rust.md",
        );
        rag.compute_graph_scores();

        let results = rag.query("Rust programming", 2);
        let context = rag.build_context(&results, true, 4000);
        assert!(!context.is_empty());
        assert!(context.contains("rust.md"));
    }

    #[test]
    fn test_community_summary() {
        let mut rag = make_rag();
        rag.ingest_text(
            "Python is great for Data Science. \
             JavaScript dominates Web Development. \
             Rust excels at Systems Programming.",
            "langs.md",
        );
        rag.compute_graph_scores();

        let communities = rag.get_community_summary();
        assert!(!communities.is_empty());
    }

    #[test]
    fn test_rrf_fusion() {
        let vector = vec![
            ("a".to_string(), 0.9f32),
            ("b".to_string(), 0.8),
            ("c".to_string(), 0.7),
        ];
        let graph = vec![
            ("b".to_string(), 5.0f64),
            ("d".to_string(), 3.0),
            ("a".to_string(), 1.0),
        ];

        let fused = rrf_fusion(&vector, &graph, 60.0);

        // "a" and "b" should be top since they appear in both lists
        let top_ids: Vec<&str> = fused.iter().take(2).map(|(id, _)| id.as_str()).collect();
        assert!(top_ids.contains(&"a"));
        assert!(top_ids.contains(&"b"));
    }
}
