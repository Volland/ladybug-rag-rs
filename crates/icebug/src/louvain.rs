use icebug_sys::*;
use std::ptr::NonNull;

use crate::Graph;

/// Louvain community detection (Parallel Louvain Method).
pub struct Louvain {
    ptr: NonNull<IcebugLouvain>,
}

impl Louvain {
    /// Create a Louvain instance.
    ///
    /// - `refine`: whether to use refinement phase
    /// - `gamma`: resolution parameter (1.0 = standard modularity)
    /// - `max_iter`: maximum number of iterations (0 = unlimited)
    pub fn new(graph: &Graph, refine: bool, gamma: f64, max_iter: u64) -> Self {
        let ptr = unsafe { icebug_louvain_new(graph.ptr.as_ptr(), refine, gamma, max_iter) };
        Self {
            ptr: NonNull::new(ptr).expect("icebug_louvain_new returned null"),
        }
    }

    /// Run the algorithm.
    pub fn run(&mut self) {
        unsafe { icebug_louvain_run(self.ptr.as_ptr()) }
    }

    /// Get the resulting partition (community assignment).
    pub fn get_partition(&self) -> Partition {
        let ptr = unsafe { icebug_louvain_get_partition(self.ptr.as_ptr()) };
        Partition {
            ptr: NonNull::new(ptr).expect("icebug_louvain_get_partition returned null"),
        }
    }
}

impl Drop for Louvain {
    fn drop(&mut self) {
        unsafe { icebug_louvain_free(self.ptr.as_ptr()) }
    }
}

/// A partition of nodes into communities.
pub struct Partition {
    ptr: NonNull<IcebugPartition>,
}

impl Partition {
    /// Get the community ID for a specific node.
    pub fn subset_of(&self, node: u64) -> u64 {
        unsafe { icebug_partition_subset_of(self.ptr.as_ptr(), node) }
    }

    /// Get the number of communities.
    pub fn number_of_subsets(&self) -> u64 {
        unsafe { icebug_partition_number_of_subsets(self.ptr.as_ptr()) }
    }

    /// Get community assignments for all nodes as a vector indexed by node ID.
    pub fn to_vec(&self, n: u64) -> Vec<u64> {
        let mut buf = vec![0u64; n as usize];
        let written = unsafe {
            icebug_partition_get_vector(self.ptr.as_ptr(), buf.as_mut_ptr(), n)
        };
        buf.truncate(written as usize);
        buf
    }
}

impl Drop for Partition {
    fn drop(&mut self) {
        unsafe { icebug_partition_free(self.ptr.as_ptr()) }
    }
}
