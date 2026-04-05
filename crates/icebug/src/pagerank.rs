use icebug_sys::*;
use std::ptr::NonNull;

use crate::Graph;

/// PageRank centrality algorithm.
pub struct PageRank {
    ptr: NonNull<IcebugPageRank>,
    node_count: u64,
}

impl PageRank {
    /// Create a PageRank instance for the given graph.
    ///
    /// - `damping`: damping factor (typically 0.85)
    /// - `tolerance`: convergence tolerance (e.g. 1e-6)
    pub fn new(graph: &Graph, damping: f64, tolerance: f64) -> Self {
        let ptr = unsafe { icebug_pagerank_new(graph.ptr.as_ptr(), damping, tolerance) };
        Self {
            ptr: NonNull::new(ptr).expect("icebug_pagerank_new returned null"),
            node_count: graph.upper_node_id_bound(),
        }
    }

    /// Run the algorithm.
    pub fn run(&mut self) {
        unsafe { icebug_pagerank_run(self.ptr.as_ptr()) }
    }

    /// Get the PageRank score for a specific node.
    pub fn score(&mut self, node: u64) -> f64 {
        unsafe { icebug_pagerank_score(self.ptr.as_ptr(), node) }
    }

    /// Get all PageRank scores as a vector indexed by node ID.
    pub fn scores(&mut self) -> Vec<f64> {
        let mut buf = vec![0.0f64; self.node_count as usize];
        let n = unsafe {
            icebug_pagerank_scores(self.ptr.as_ptr(), buf.as_mut_ptr(), self.node_count)
        };
        buf.truncate(n as usize);
        buf
    }
}

impl Drop for PageRank {
    fn drop(&mut self) {
        unsafe { icebug_pagerank_free(self.ptr.as_ptr()) }
    }
}
