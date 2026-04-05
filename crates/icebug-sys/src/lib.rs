//! Raw FFI bindings to the icebug C wrapper.
//!
//! These are unsafe, low-level bindings. Use the `icebug` crate for a safe API.

#![allow(non_camel_case_types)]

use std::os::raw::c_void;

// Opaque handles
#[repr(C)]
pub struct IcebugGraph {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct IcebugPageRank {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct IcebugLouvain {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct IcebugPartition {
    _opaque: [u8; 0],
}
#[repr(C)]
pub struct IcebugBFS {
    _opaque: [u8; 0],
}

pub type IcebugEdgeCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, u: u64, v: u64, weight: f64, edge_id: u64)>;
pub type IcebugNodeCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, node: u64)>;

extern "C" {
    // --- Graph lifecycle ---
    pub fn icebug_graph_new(n: u64, weighted: bool, directed: bool) -> *mut IcebugGraph;
    pub fn icebug_graph_free(g: *mut IcebugGraph);

    // --- Graph mutation ---
    pub fn icebug_graph_add_node(g: *mut IcebugGraph) -> u64;
    pub fn icebug_graph_add_nodes(g: *mut IcebugGraph, count: u64);
    pub fn icebug_graph_add_edge(g: *mut IcebugGraph, u: u64, v: u64, weight: f64);
    pub fn icebug_graph_remove_edge(g: *mut IcebugGraph, u: u64, v: u64);
    pub fn icebug_graph_remove_node(g: *mut IcebugGraph, u: u64);
    pub fn icebug_graph_set_weight(g: *mut IcebugGraph, u: u64, v: u64, w: f64);

    // --- Graph queries ---
    pub fn icebug_graph_number_of_nodes(g: *const IcebugGraph) -> u64;
    pub fn icebug_graph_number_of_edges(g: *const IcebugGraph) -> u64;
    pub fn icebug_graph_has_edge(g: *const IcebugGraph, u: u64, v: u64) -> bool;
    pub fn icebug_graph_weight(g: *const IcebugGraph, u: u64, v: u64) -> f64;
    pub fn icebug_graph_degree(g: *const IcebugGraph, u: u64) -> u64;
    pub fn icebug_graph_degree_in(g: *const IcebugGraph, u: u64) -> u64;
    pub fn icebug_graph_degree_out(g: *const IcebugGraph, u: u64) -> u64;
    pub fn icebug_graph_is_weighted(g: *const IcebugGraph) -> bool;
    pub fn icebug_graph_is_directed(g: *const IcebugGraph) -> bool;
    pub fn icebug_graph_has_node(g: *const IcebugGraph, u: u64) -> bool;
    pub fn icebug_graph_upper_node_id_bound(g: *const IcebugGraph) -> u64;

    // --- Neighbor access ---
    pub fn icebug_graph_neighbors(g: *const IcebugGraph, u: u64, out: *mut u64, out_len: u64) -> u64;
    pub fn icebug_graph_in_neighbors(g: *const IcebugGraph, u: u64, out: *mut u64, out_len: u64) -> u64;

    // --- Iteration ---
    pub fn icebug_graph_for_edges(g: *const IcebugGraph, callback: IcebugEdgeCallback, user_data: *mut c_void);
    pub fn icebug_graph_for_nodes(g: *const IcebugGraph, callback: IcebugNodeCallback, user_data: *mut c_void);

    // --- PageRank ---
    pub fn icebug_pagerank_new(g: *const IcebugGraph, damping: f64, tolerance: f64) -> *mut IcebugPageRank;
    pub fn icebug_pagerank_run(pr: *mut IcebugPageRank);
    pub fn icebug_pagerank_score(pr: *mut IcebugPageRank, node: u64) -> f64;
    pub fn icebug_pagerank_scores(pr: *mut IcebugPageRank, out: *mut f64, out_len: u64) -> u64;
    pub fn icebug_pagerank_free(pr: *mut IcebugPageRank);

    // --- Louvain ---
    pub fn icebug_louvain_new(g: *const IcebugGraph, refine: bool, gamma: f64, max_iter: u64) -> *mut IcebugLouvain;
    pub fn icebug_louvain_run(lv: *mut IcebugLouvain);
    pub fn icebug_louvain_get_partition(lv: *const IcebugLouvain) -> *mut IcebugPartition;
    pub fn icebug_louvain_free(lv: *mut IcebugLouvain);

    // --- Partition ---
    pub fn icebug_partition_subset_of(p: *const IcebugPartition, node: u64) -> u64;
    pub fn icebug_partition_number_of_subsets(p: *const IcebugPartition) -> u64;
    pub fn icebug_partition_get_vector(p: *const IcebugPartition, out: *mut u64, out_len: u64) -> u64;
    pub fn icebug_partition_free(p: *mut IcebugPartition);

    // --- BFS ---
    pub fn icebug_bfs_new(g: *const IcebugGraph, source: u64, store_paths: bool) -> *mut IcebugBFS;
    pub fn icebug_bfs_run(bfs: *mut IcebugBFS);
    pub fn icebug_bfs_distance(bfs: *mut IcebugBFS, target: u64) -> f64;
    pub fn icebug_bfs_distances(bfs: *mut IcebugBFS, out: *mut f64, out_len: u64) -> u64;
    pub fn icebug_bfs_free(bfs: *mut IcebugBFS);
}
