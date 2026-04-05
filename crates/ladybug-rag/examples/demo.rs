//! Demo: Hybrid Graph RAG with icebug graph algorithms
//!
//! Run with: cargo run --example demo

use ladybug_rag::embeddings::SimpleEmbedder;
use ladybug_rag::rag::{HybridGraphRag, RagConfig};

fn main() {
    println!("=== Ladybug Hybrid Graph RAG Demo ===\n");

    // 1. Initialize the RAG engine
    let mut rag = HybridGraphRag::new(
        Box::new(SimpleEmbedder::default()),
        RagConfig {
            chunk_size: 512,
            chunk_overlap: 100,
            ..RagConfig::default()
        },
    );

    // 2. Ingest documents
    let docs = vec![
        (include_str!("data/rust_overview.txt"), "rust_overview.md"),
        (include_str!("data/graph_algorithms.txt"), "graph_algorithms.md"),
    ];

    println!("Ingesting {} documents...", docs.len());
    for (text, source) in &docs {
        rag.ingest_text(text, source);
        println!("  - Ingested: {}", source);
    }

    // 3. Compute graph scores (PageRank + Louvain communities via icebug)
    println!("\nComputing graph scores (PageRank + Louvain)...");
    rag.compute_graph_scores();

    let stats = rag.stats();
    println!("Knowledge base statistics:");
    println!("  Chunks:      {}", stats.chunk_count);
    println!("  Entities:    {}", stats.entity_count);
    println!("  Mentions:    {}", stats.mention_count);
    println!("  Relations:   {}", stats.relation_count);
    println!("  Communities: {}", stats.community_count);

    // 4. Show community structure
    println!("\nCommunity summary:");
    let communities = rag.get_community_summary();
    for (community_id, members) in &communities {
        println!(
            "  Community {}: {} ({})",
            community_id,
            members.join(", "),
            members.len()
        );
    }

    // 5. Query the knowledge base
    let queries = vec![
        "How does Rust ensure memory safety?",
        "What is PageRank used for?",
        "How do graph algorithms help with community detection?",
    ];

    for query in queries {
        println!("\n--- Query: \"{}\" ---", query);
        let results = rag.query(query, 3);

        for (i, result) in results.iter().enumerate() {
            println!(
                "  [{}] score={:.4} method={:?} source={}",
                i + 1,
                result.score,
                result.method,
                result.chunk.source
            );
            // Show first 120 chars of the chunk
            let preview: String = result.chunk.text.chars().take(120).collect();
            println!("      \"{}...\"", preview);
            if !result.entities.is_empty() {
                println!("      entities: {}", result.entities.join(", "));
            }
        }

        // Build LLM-ready context
        let context = rag.build_context(&results, true, 2000);
        println!("\n  Context ({} chars):", context.len());
        // Show first 200 chars
        let ctx_preview: String = context.chars().take(200).collect();
        println!("  {}", ctx_preview);
    }

    println!("\n=== Demo Complete ===");
}
