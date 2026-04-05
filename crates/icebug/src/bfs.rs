use icebug_sys::*;
use std::ptr::NonNull;

use crate::Graph;

/// Breadth-first search from a source node.
pub struct Bfs {
    ptr: NonNull<IcebugBFS>,
    node_count: u64,
}

impl Bfs {
    /// Create a BFS instance.
    ///
    /// - `source`: source node ID
    /// - `store_paths`: whether to store shortest-path predecessors
    pub fn new(graph: &Graph, source: u64, store_paths: bool) -> Self {
        let ptr = unsafe { icebug_bfs_new(graph.ptr.as_ptr(), source, store_paths) };
        Self {
            ptr: NonNull::new(ptr).expect("icebug_bfs_new returned null"),
            node_count: graph.upper_node_id_bound(),
        }
    }

    /// Run BFS.
    pub fn run(&mut self) {
        unsafe { icebug_bfs_run(self.ptr.as_ptr()) }
    }

    /// Get the distance from source to `target`.
    pub fn distance(&mut self, target: u64) -> f64 {
        unsafe { icebug_bfs_distance(self.ptr.as_ptr(), target) }
    }

    /// Get all distances from the source node.
    pub fn distances(&mut self) -> Vec<f64> {
        let mut buf = vec![0.0f64; self.node_count as usize];
        let n = unsafe {
            icebug_bfs_distances(self.ptr.as_ptr(), buf.as_mut_ptr(), self.node_count)
        };
        buf.truncate(n as usize);
        buf
    }
}

impl Drop for Bfs {
    fn drop(&mut self) {
        unsafe { icebug_bfs_free(self.ptr.as_ptr()) }
    }
}
