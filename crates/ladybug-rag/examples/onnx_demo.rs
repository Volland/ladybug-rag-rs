//! Demo: Hybrid Graph RAG with real ONNX multilingual embeddings
//!
//! Downloads the model automatically from HuggingFace Hub on first run.
//!
//! Run with:
//!   cargo run -p ladybug-rag --features onnx --example onnx_demo

use ladybug_rag::embeddings::Embedder;
use ladybug_rag::onnx_embedder::OnnxEmbedder;
use ladybug_rag::rag::{HybridGraphRag, RagConfig};

fn main() {
    println!("=== Hybrid Graph RAG with ONNX Multilingual Embeddings ===\n");

    // 1. Load the multilingual embedding model
    // This downloads ~130MB on first run and caches it in ~/.cache/huggingface/
    let model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
    println!("Loading model: {}", model_name);
    println!("(first run downloads ~130MB from HuggingFace Hub)\n");

    let embedder = OnnxEmbedder::from_hf_hub(model_name, 384)
        .expect("Failed to load ONNX model");

    println!("Model loaded! Dimension: {}\n", embedder.dimension());

    // 2. Initialize RAG with real embeddings
    let mut rag = HybridGraphRag::new(
        Box::new(embedder),
        RagConfig {
            chunk_size: 512,
            chunk_overlap: 100,
            ..RagConfig::default()
        },
    );

    // 3. Ingest multilingual documents
    let docs = vec![
        (
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It achieves memory safety without garbage collection through its ownership system. \
             The borrow checker enforces strict rules at compile time, preventing data races.",
            "rust_en.md",
        ),
        (
            "Rust ist eine Systemprogrammiersprache, die sich auf Sicherheit, Geschwindigkeit \
             und Nebenläufigkeit konzentriert. Sie erreicht Speichersicherheit ohne Garbage \
             Collection durch ihr Ownership-System.",
            "rust_de.md",
        ),
        (
            "Rust — це системна мова програмування, орієнтована на безпеку, швидкість та \
             паралелізм. Вона досягає безпеки пам'яті без збирача сміття завдяки системі \
             володіння (ownership).",
            "rust_uk.md",
        ),
        (
            "Graph algorithms like PageRank and Louvain community detection are essential \
             for analyzing network structures. PageRank measures node importance based on \
             the link structure, while Louvain identifies densely connected communities.",
            "graph_en.md",
        ),
        (
            "Les algorithmes de graphes comme PageRank et la détection de communautés Louvain \
             sont essentiels pour analyser les structures de réseau. PageRank mesure l'importance \
             des noeuds basée sur la structure des liens.",
            "graph_fr.md",
        ),
    ];

    println!("Ingesting {} multilingual documents...", docs.len());
    for (text, source) in &docs {
        rag.ingest_text(text, source);
        println!("  - {}", source);
    }

    // 4. Compute graph scores
    println!("\nComputing graph scores (PageRank + Louvain)...");
    rag.compute_graph_scores();

    let stats = rag.stats();
    println!("Knowledge base:");
    println!("  Chunks:      {}", stats.chunk_count);
    println!("  Entities:    {}", stats.entity_count);
    println!("  Communities: {}", stats.community_count);

    // 5. Query in different languages
    let queries = vec![
        ("English", "How does Rust ensure memory safety?"),
        ("German", "Wie funktioniert das Ownership-System in Rust?"),
        ("Ukrainian", "Що таке система володіння в Rust?"),
        ("English", "What is PageRank used for in graph analysis?"),
        ("French", "Comment fonctionne la détection de communautés?"),
    ];

    for (lang, query) in queries {
        println!("\n--- Query [{}]: \"{}\" ---", lang, query);
        let results = rag.query(query, 3);

        for (i, result) in results.iter().enumerate() {
            let preview: String = result.chunk.text.chars().take(80).collect();
            println!(
                "  [{}] score={:.4} method={:?} source={}",
                i + 1,
                result.score,
                result.method,
                result.chunk.source
            );
            println!("      \"{}...\"", preview);
        }
    }

    println!("\n=== Demo Complete ===");
    println!("\nThe multilingual model correctly retrieves relevant content");
    println!("across languages — a German query finds English+German chunks about Rust,");
    println!("and a French query finds English+French chunks about graph algorithms.");
}
