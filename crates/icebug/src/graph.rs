use icebug_sys::*;
use std::os::raw::c_void;
use std::ptr::NonNull;

/// A graph data structure backed by icebug (NetworKit).
///
/// Supports weighted/unweighted and directed/undirected graphs.
pub struct Graph {
    pub(crate) ptr: NonNull<IcebugGraph>,
}

unsafe impl Send for Graph {}

impl Graph {
    /// Create a new graph with `n` initial nodes.
    pub fn new(n: u64, weighted: bool, directed: bool) -> Self {
        let ptr = unsafe { icebug_graph_new(n, weighted, directed) };
        Self {
            ptr: NonNull::new(ptr).expect("icebug_graph_new returned null"),
        }
    }

    /// Add a node and return its ID.
    pub fn add_node(&mut self) -> u64 {
        unsafe { icebug_graph_add_node(self.ptr.as_ptr()) }
    }

    /// Add `count` new nodes.
    pub fn add_nodes(&mut self, count: u64) {
        unsafe { icebug_graph_add_nodes(self.ptr.as_ptr(), count) }
    }

    /// Add an edge between u and v with a given weight.
    pub fn add_edge(&mut self, u: u64, v: u64, weight: f64) {
        unsafe { icebug_graph_add_edge(self.ptr.as_ptr(), u, v, weight) }
    }

    /// Remove an edge.
    pub fn remove_edge(&mut self, u: u64, v: u64) {
        unsafe { icebug_graph_remove_edge(self.ptr.as_ptr(), u, v) }
    }

    /// Remove a node.
    pub fn remove_node(&mut self, u: u64) {
        unsafe { icebug_graph_remove_node(self.ptr.as_ptr(), u) }
    }

    /// Set the weight of an edge.
    pub fn set_weight(&mut self, u: u64, v: u64, w: f64) {
        unsafe { icebug_graph_set_weight(self.ptr.as_ptr(), u, v, w) }
    }

    pub fn number_of_nodes(&self) -> u64 {
        unsafe { icebug_graph_number_of_nodes(self.ptr.as_ptr()) }
    }

    pub fn number_of_edges(&self) -> u64 {
        unsafe { icebug_graph_number_of_edges(self.ptr.as_ptr()) }
    }

    pub fn has_edge(&self, u: u64, v: u64) -> bool {
        unsafe { icebug_graph_has_edge(self.ptr.as_ptr(), u, v) }
    }

    pub fn weight(&self, u: u64, v: u64) -> f64 {
        unsafe { icebug_graph_weight(self.ptr.as_ptr(), u, v) }
    }

    pub fn degree(&self, u: u64) -> u64 {
        unsafe { icebug_graph_degree(self.ptr.as_ptr(), u) }
    }

    pub fn degree_in(&self, u: u64) -> u64 {
        unsafe { icebug_graph_degree_in(self.ptr.as_ptr(), u) }
    }

    pub fn degree_out(&self, u: u64) -> u64 {
        unsafe { icebug_graph_degree_out(self.ptr.as_ptr(), u) }
    }

    pub fn is_weighted(&self) -> bool {
        unsafe { icebug_graph_is_weighted(self.ptr.as_ptr()) }
    }

    pub fn is_directed(&self) -> bool {
        unsafe { icebug_graph_is_directed(self.ptr.as_ptr()) }
    }

    pub fn has_node(&self, u: u64) -> bool {
        unsafe { icebug_graph_has_node(self.ptr.as_ptr(), u) }
    }

    pub fn upper_node_id_bound(&self) -> u64 {
        unsafe { icebug_graph_upper_node_id_bound(self.ptr.as_ptr()) }
    }

    /// Get the neighbors of node `u`.
    pub fn neighbors(&self, u: u64) -> Vec<u64> {
        let deg = self.degree(u) as usize;
        if deg == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u64; deg];
        let n = unsafe {
            icebug_graph_neighbors(self.ptr.as_ptr(), u, buf.as_mut_ptr(), deg as u64)
        };
        buf.truncate(n as usize);
        buf
    }

    /// Get the in-neighbors of node `u` (for directed graphs).
    pub fn in_neighbors(&self, u: u64) -> Vec<u64> {
        let deg = self.degree_in(u) as usize;
        if deg == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u64; deg];
        let n = unsafe {
            icebug_graph_in_neighbors(self.ptr.as_ptr(), u, buf.as_mut_ptr(), deg as u64)
        };
        buf.truncate(n as usize);
        buf
    }

    /// Iterate over all edges, calling `f(u, v, weight, edge_id)`.
    pub fn for_edges<F: FnMut(u64, u64, f64, u64)>(&self, mut f: F) {
        unsafe extern "C" fn trampoline<F: FnMut(u64, u64, f64, u64)>(
            user_data: *mut c_void,
            u: u64,
            v: u64,
            weight: f64,
            edge_id: u64,
        ) {
            let f = &mut *(user_data as *mut F);
            f(u, v, weight, edge_id);
        }

        unsafe {
            icebug_graph_for_edges(
                self.ptr.as_ptr(),
                Some(trampoline::<F>),
                &mut f as *mut F as *mut c_void,
            );
        }
    }

    /// Iterate over all nodes, calling `f(node_id)`.
    pub fn for_nodes<F: FnMut(u64)>(&self, mut f: F) {
        unsafe extern "C" fn trampoline<F: FnMut(u64)>(
            user_data: *mut c_void,
            node: u64,
        ) {
            let f = &mut *(user_data as *mut F);
            f(node);
        }

        unsafe {
            icebug_graph_for_nodes(
                self.ptr.as_ptr(),
                Some(trampoline::<F>),
                &mut f as *mut F as *mut c_void,
            );
        }
    }

    /// Collect all node IDs.
    pub fn nodes(&self) -> Vec<u64> {
        let mut result = Vec::with_capacity(self.number_of_nodes() as usize);
        self.for_nodes(|n| result.push(n));
        result
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { icebug_graph_free(self.ptr.as_ptr()) }
    }
}
