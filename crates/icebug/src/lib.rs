//! Safe Rust bindings for the icebug graph analysis library.
//!
//! Provides Graph, PageRank, Louvain community detection, and BFS.

mod graph;
mod pagerank;
mod louvain;
mod bfs;

pub use graph::Graph;
pub use pagerank::PageRank;
pub use louvain::{Louvain, Partition};
pub use bfs::Bfs;
