use std::collections::HashMap;

use icebug::{Graph, Louvain, PageRank};

/// Manages the entity/chunk knowledge graph backed by icebug.
///
/// Maps string IDs to icebug node IDs and stores relationships.
pub struct GraphStore {
    graph: Graph,
    /// Map from string ID -> icebug node ID
    id_to_node: HashMap<String, u64>,
    /// Map from icebug node ID -> string ID
    node_to_id: HashMap<u64, String>,
    /// Track which node IDs are chunks vs entities
    chunk_nodes: HashMap<String, u64>,
    entity_nodes: HashMap<String, u64>,
    /// MENTIONS edges: chunk -> entity
    mention_edges: Vec<(String, String)>,
    /// RELATES_TO edges: entity -> entity
    relation_edges: Vec<(String, String)>,
    /// NEXT_CHUNK edges: chunk -> chunk (sequential)
    next_chunk_edges: Vec<(String, String)>,
}

impl GraphStore {
    pub fn new() -> Self {
        Self {
            graph: Graph::new(0, true, false),
            id_to_node: HashMap::new(),
            node_to_id: HashMap::new(),
            chunk_nodes: HashMap::new(),
            entity_nodes: HashMap::new(),
            mention_edges: Vec::new(),
            relation_edges: Vec::new(),
            next_chunk_edges: Vec::new(),
        }
    }

    fn ensure_node(&mut self, id: &str) -> u64 {
        if let Some(&node) = self.id_to_node.get(id) {
            return node;
        }
        let node = self.graph.add_node();
        self.id_to_node.insert(id.to_string(), node);
        self.node_to_id.insert(node, id.to_string());
        node
    }

    /// Register a chunk node.
    pub fn add_chunk(&mut self, chunk_id: &str) -> u64 {
        let node = self.ensure_node(chunk_id);
        self.chunk_nodes.insert(chunk_id.to_string(), node);
        node
    }

    /// Register an entity node.
    pub fn add_entity(&mut self, entity_id: &str) -> u64 {
        let node = self.ensure_node(entity_id);
        self.entity_nodes.insert(entity_id.to_string(), node);
        node
    }

    /// Add a MENTIONS edge (chunk -> entity).
    pub fn add_mention(&mut self, chunk_id: &str, entity_id: &str) {
        let u = self.ensure_node(chunk_id);
        let v = self.ensure_node(entity_id);
        if !self.graph.has_edge(u, v) {
            self.graph.add_edge(u, v, 1.0);
            self.mention_edges
                .push((chunk_id.to_string(), entity_id.to_string()));
        }
    }

    /// Add a RELATES_TO edge (entity -> entity).
    pub fn add_relation(&mut self, source_id: &str, target_id: &str, weight: f64) {
        let u = self.ensure_node(source_id);
        let v = self.ensure_node(target_id);
        if !self.graph.has_edge(u, v) {
            self.graph.add_edge(u, v, weight);
            self.relation_edges
                .push((source_id.to_string(), target_id.to_string()));
        }
    }

    /// Add a NEXT_CHUNK edge for sequential ordering.
    pub fn add_next_chunk(&mut self, from_id: &str, to_id: &str) {
        let u = self.ensure_node(from_id);
        let v = self.ensure_node(to_id);
        if !self.graph.has_edge(u, v) {
            self.graph.add_edge(u, v, 0.5);
            self.next_chunk_edges
                .push((from_id.to_string(), to_id.to_string()));
        }
    }

    /// Compute PageRank scores for all nodes.
    pub fn compute_pagerank(&self, damping: f64, tolerance: f64) -> HashMap<String, f64> {
        if self.graph.number_of_nodes() == 0 {
            return HashMap::new();
        }
        let mut pr = PageRank::new(&self.graph, damping, tolerance);
        pr.run();

        let scores = pr.scores();
        let mut result = HashMap::new();
        for (&node, id) in &self.node_to_id {
            if (node as usize) < scores.len() {
                result.insert(id.clone(), scores[node as usize]);
            }
        }
        result
    }

    /// Run Louvain community detection.
    /// Returns a map of string ID -> community ID.
    pub fn compute_communities(&self, gamma: f64) -> (HashMap<String, u64>, u64) {
        if self.graph.number_of_nodes() == 0 {
            return (HashMap::new(), 0);
        }
        let mut lv = Louvain::new(&self.graph, false, gamma, 32);
        lv.run();
        let partition = lv.get_partition();
        let num_communities = partition.number_of_subsets();
        let assignments = partition.to_vec(self.graph.upper_node_id_bound());

        let mut result = HashMap::new();
        for (&node, id) in &self.node_to_id {
            if (node as usize) < assignments.len() {
                result.insert(id.clone(), assignments[node as usize]);
            }
        }
        (result, num_communities)
    }

    /// Expand from seed entity IDs via graph traversal.
    /// Returns chunk IDs reachable within `hops` from the seed entities,
    /// weighted by PageRank scores.
    pub fn graph_expand(
        &self,
        seed_entity_ids: &[String],
        hops: usize,
        pagerank_scores: &HashMap<String, f64>,
    ) -> Vec<(String, f64)> {
        let mut visited = std::collections::HashSet::new();
        let mut frontier: Vec<u64> = Vec::new();

        // Start from seed entities
        for eid in seed_entity_ids {
            if let Some(&node) = self.id_to_node.get(eid) {
                frontier.push(node);
                visited.insert(node);
            }
        }

        let mut discovered_chunks: HashMap<String, f64> = HashMap::new();

        for _hop in 0..hops {
            let mut next_frontier = Vec::new();
            for &node in &frontier {
                let neighbors = self.graph.neighbors(node);
                for neighbor in neighbors {
                    if visited.insert(neighbor) {
                        next_frontier.push(neighbor);
                    }

                    // If neighbor is a chunk, record it
                    if let Some(id) = self.node_to_id.get(&neighbor) {
                        if self.chunk_nodes.contains_key(id) {
                            let pr_score = pagerank_scores
                                .get(id)
                                .copied()
                                .unwrap_or(0.0);
                            let current = discovered_chunks.entry(id.clone()).or_insert(0.0);
                            *current += 1.0 + pr_score * 10.0;
                        }
                    }
                }
            }
            frontier = next_frontier;
        }

        let mut results: Vec<(String, f64)> = discovered_chunks.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Get entity IDs connected to a chunk via MENTIONS edges.
    pub fn chunk_entities(&self, chunk_id: &str) -> Vec<String> {
        self.mention_edges
            .iter()
            .filter(|(c, _)| c == chunk_id)
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// Get neighboring chunk IDs (via NEXT_CHUNK).
    pub fn neighboring_chunks(&self, chunk_id: &str) -> Vec<String> {
        let mut neighbors = Vec::new();
        for (from, to) in &self.next_chunk_edges {
            if from == chunk_id {
                neighbors.push(to.clone());
            }
            if to == chunk_id {
                neighbors.push(from.clone());
            }
        }
        neighbors
    }

    pub fn mention_count(&self) -> usize {
        self.mention_edges.len()
    }

    pub fn relation_count(&self) -> usize {
        self.relation_edges.len()
    }

    pub fn entity_count(&self) -> usize {
        self.entity_nodes.len()
    }
}

impl Default for GraphStore {
    fn default() -> Self {
        Self::new()
    }
}
