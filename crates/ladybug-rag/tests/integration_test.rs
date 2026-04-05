use ladybug_rag::embeddings::SimpleEmbedder;
use ladybug_rag::rag::{HybridGraphRag, RagConfig};

const RUST_DOC: &str = "\
Rust is a systems programming language focused on safety, speed, and concurrency. \
It achieves memory safety without garbage collection through its ownership system. \
The Rust compiler enforces strict borrowing rules at compile time.

Cargo is Rust's package manager and build system. \
It handles downloading dependencies, compiling packages, and running tests. \
Cargo uses Cargo.toml files to specify project configuration and dependencies.

Traits in Rust provide a way to define shared behavior. \
They are similar to interfaces in other languages. \
Generic programming in Rust relies heavily on trait bounds.

The Rust Standard Library provides essential types like Vec, String, and HashMap. \
These collections are used extensively in Rust programs. \
The standard library also includes I/O, threading, and networking primitives.";

const ML_DOC: &str = "\
Machine Learning is a subset of Artificial Intelligence. \
Deep Learning uses Neural Networks with multiple layers. \
Convolutional Neural Networks are used for Image Recognition tasks.

Natural Language Processing enables computers to understand human language. \
Transformer models have revolutionized Natural Language Processing. \
Large Language Models like GPT are trained on massive text corpora.

Supervised Learning requires labeled training data. \
Classification and Regression are common Supervised Learning tasks. \
Decision Trees and Random Forests are popular classification algorithms.

Unsupervised Learning discovers patterns without labels. \
Clustering algorithms group similar data points together. \
K-Means and DBSCAN are widely used Clustering methods.";

fn make_rag() -> HybridGraphRag {
    HybridGraphRag::new(
        Box::new(SimpleEmbedder::default()),
        RagConfig {
            chunk_size: 300,
            chunk_overlap: 50,
            ..RagConfig::default()
        },
    )
}

#[test]
fn test_multi_document_ingestion() {
    let mut rag = make_rag();
    rag.ingest_documents(&[(RUST_DOC, "rust.md"), (ML_DOC, "ml.md")]);
    rag.compute_graph_scores();

    let stats = rag.stats();
    assert!(stats.chunk_count >= 4, "Expected at least 4 chunks, got {}", stats.chunk_count);
    assert!(stats.entity_count >= 5, "Expected at least 5 entities, got {}", stats.entity_count);
    assert!(stats.mention_count > 0, "Expected mentions");
    assert!(stats.community_count >= 1, "Expected at least 1 community");
}

#[test]
fn test_query_relevance() {
    let mut rag = make_rag();
    rag.ingest_documents(&[(RUST_DOC, "rust.md"), (ML_DOC, "ml.md")]);
    rag.compute_graph_scores();

    // Query about Rust should return Rust-related chunks
    let results = rag.query("What is Rust's ownership system?", 3);
    assert!(!results.is_empty());

    let has_rust_source = results.iter().any(|r| r.chunk.source == "rust.md");
    assert!(has_rust_source, "Query about Rust should return rust.md chunks");

    // Query about ML should return ML-related chunks
    let ml_results = rag.query("How does Deep Learning work?", 3);
    assert!(!ml_results.is_empty());

    let has_ml_source = ml_results.iter().any(|r| r.chunk.source == "ml.md");
    assert!(has_ml_source, "Query about ML should return ml.md chunks");
}

#[test]
fn test_hybrid_retrieval_methods() {
    let mut rag = make_rag();
    rag.ingest_documents(&[(RUST_DOC, "rust.md"), (ML_DOC, "ml.md")]);
    rag.compute_graph_scores();

    let results = rag.query("Neural Networks and Deep Learning", 5);
    assert!(!results.is_empty());

    // Results should have scores > 0
    for r in &results {
        assert!(r.score > 0.0, "All results should have positive scores");
    }

    // Results should be sorted by score descending
    for w in results.windows(2) {
        assert!(w[0].score >= w[1].score, "Results should be sorted by score");
    }
}

#[test]
fn test_context_building() {
    let mut rag = make_rag();
    rag.ingest_text(RUST_DOC, "rust.md");
    rag.compute_graph_scores();

    let results = rag.query("Cargo build system", 3);
    let context = rag.build_context(&results, true, 2000);

    assert!(!context.is_empty());
    assert!(context.contains("rust.md"), "Context should include source");
    assert!(context.contains("score:"), "Context should include scores");
}

#[test]
fn test_context_max_chars_respected() {
    let mut rag = make_rag();
    rag.ingest_text(RUST_DOC, "rust.md");
    rag.ingest_text(ML_DOC, "ml.md");
    rag.compute_graph_scores();

    let results = rag.query("programming", 10);
    let context = rag.build_context(&results, false, 500);

    assert!(
        context.len() <= 600, // Allow small margin for last chunk
        "Context should respect max_chars limit: got {} chars",
        context.len()
    );
}

#[test]
fn test_community_summary_structure() {
    let mut rag = make_rag();
    rag.ingest_documents(&[(RUST_DOC, "rust.md"), (ML_DOC, "ml.md")]);
    rag.compute_graph_scores();

    let communities = rag.get_community_summary();
    assert!(!communities.is_empty(), "Should have at least one community");

    let total_entities: usize = communities.values().map(|v| v.len()).sum();
    assert!(total_entities > 0, "Communities should contain entities");
}

#[test]
fn test_entity_extraction_in_results() {
    let mut rag = make_rag();
    rag.ingest_text(ML_DOC, "ml.md");
    rag.compute_graph_scores();

    let results = rag.query("Neural Networks", 3);
    assert!(!results.is_empty());

    // At least some results should have associated entities
    let has_entities = results.iter().any(|r| !r.entities.is_empty());
    assert!(has_entities, "Results should include extracted entities");
}

#[test]
fn test_empty_query_handling() {
    let mut rag = make_rag();
    rag.ingest_text("Some text here.", "doc.md");
    rag.compute_graph_scores();

    // Should not panic on empty-ish queries
    let results = rag.query("", 3);
    // Empty query may or may not return results, but shouldn't panic
    let _ = results;
}
