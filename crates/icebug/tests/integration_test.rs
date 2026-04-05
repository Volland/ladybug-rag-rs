use icebug::{Bfs, Graph, Louvain, PageRank};

#[test]
fn test_graph_basic_operations() {
    let mut g = Graph::new(5, false, false);

    assert_eq!(g.number_of_nodes(), 5);
    assert_eq!(g.number_of_edges(), 0);

    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(2, 3, 1.0);
    g.add_edge(3, 4, 1.0);

    assert_eq!(g.number_of_edges(), 4);
    assert!(g.has_edge(0, 1));
    assert!(!g.has_edge(0, 3));
    assert_eq!(g.degree(1), 2); // connected to 0 and 2
}

#[test]
fn test_graph_add_remove_nodes() {
    let mut g = Graph::new(0, false, false);
    let n0 = g.add_node();
    let n1 = g.add_node();
    let n2 = g.add_node();

    assert_eq!(g.number_of_nodes(), 3);
    g.add_edge(n0, n1, 1.0);
    g.add_edge(n1, n2, 1.0);

    assert!(g.has_node(n0));
    assert_eq!(g.number_of_edges(), 2);

    g.remove_node(n2);
    assert!(!g.has_node(n2));
    assert_eq!(g.number_of_edges(), 1);
}

#[test]
fn test_graph_weighted() {
    let mut g = Graph::new(3, true, false);
    g.add_edge(0, 1, 2.5);
    g.add_edge(1, 2, 3.7);

    assert!(g.is_weighted());
    assert!((g.weight(0, 1) - 2.5).abs() < 1e-10);
    assert!((g.weight(1, 2) - 3.7).abs() < 1e-10);

    g.set_weight(0, 1, 5.0);
    assert!((g.weight(0, 1) - 5.0).abs() < 1e-10);
}

#[test]
fn test_graph_directed() {
    let mut g = Graph::new(3, false, true);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);

    assert!(g.is_directed());
    assert!(g.has_edge(0, 1));
    assert!(!g.has_edge(1, 0));

    assert_eq!(g.degree_out(0), 1);
    assert_eq!(g.degree_in(0), 0);
    assert_eq!(g.degree_in(1), 1);
}

#[test]
fn test_graph_neighbors() {
    let mut g = Graph::new(4, false, false);
    g.add_edge(0, 1, 1.0);
    g.add_edge(0, 2, 1.0);
    g.add_edge(0, 3, 1.0);

    let neighbors = g.neighbors(0);
    assert_eq!(neighbors.len(), 3);
    assert!(neighbors.contains(&1));
    assert!(neighbors.contains(&2));
    assert!(neighbors.contains(&3));
}

#[test]
fn test_graph_iteration() {
    let mut g = Graph::new(3, false, false);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);

    let mut edges = Vec::new();
    g.for_edges(|u, v, _w, _eid| {
        edges.push((u, v));
    });
    assert_eq!(edges.len(), 2);

    let nodes = g.nodes();
    assert_eq!(nodes.len(), 3);
}

#[test]
fn test_pagerank_simple() {
    // Star graph: node 0 is the hub connected to 1,2,3,4
    let mut g = Graph::new(5, false, false);
    g.add_edge(0, 1, 1.0);
    g.add_edge(0, 2, 1.0);
    g.add_edge(0, 3, 1.0);
    g.add_edge(0, 4, 1.0);

    let mut pr = PageRank::new(&g, 0.85, 1e-6);
    pr.run();

    let scores = pr.scores();
    assert_eq!(scores.len(), 5);

    // Hub node should have the highest PageRank
    let hub_score = scores[0];
    for &score in &scores[1..] {
        assert!(hub_score >= score, "Hub should have highest PageRank");
    }
}

#[test]
fn test_pagerank_ring() {
    // Ring graph: all nodes should have equal PageRank
    let mut g = Graph::new(4, false, false);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(2, 3, 1.0);
    g.add_edge(3, 0, 1.0);

    let mut pr = PageRank::new(&g, 0.85, 1e-8);
    pr.run();

    let scores = pr.scores();
    let avg = scores.iter().sum::<f64>() / scores.len() as f64;
    for &s in &scores {
        assert!(
            (s - avg).abs() < 1e-4,
            "Ring graph: all PageRank scores should be roughly equal"
        );
    }
}

#[test]
fn test_louvain_two_cliques() {
    // Two cliques connected by a single edge — should detect 2 communities
    let mut g = Graph::new(8, false, false);

    // Clique 1: nodes 0-3
    for i in 0..4u64 {
        for j in (i + 1)..4 {
            g.add_edge(i, j, 1.0);
        }
    }
    // Clique 2: nodes 4-7
    for i in 4..8u64 {
        for j in (i + 1)..8 {
            g.add_edge(i, j, 1.0);
        }
    }
    // Bridge
    g.add_edge(3, 4, 1.0);

    let mut lv = Louvain::new(&g, false, 1.0, 32);
    lv.run();

    let partition = lv.get_partition();
    let num_communities = partition.number_of_subsets();
    assert!(
        num_communities >= 2,
        "Should detect at least 2 communities, got {}",
        num_communities
    );

    // Nodes in the same clique should be in the same community
    assert_eq!(partition.subset_of(0), partition.subset_of(1));
    assert_eq!(partition.subset_of(4), partition.subset_of(5));

    // Nodes in different cliques should be in different communities
    assert_ne!(partition.subset_of(0), partition.subset_of(4));
}

#[test]
fn test_bfs_distances() {
    // Linear graph: 0-1-2-3-4
    let mut g = Graph::new(5, false, false);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(2, 3, 1.0);
    g.add_edge(3, 4, 1.0);

    let mut bfs = Bfs::new(&g, 0, false);
    bfs.run();

    assert!((bfs.distance(0) - 0.0).abs() < 1e-10);
    assert!((bfs.distance(1) - 1.0).abs() < 1e-10);
    assert!((bfs.distance(2) - 2.0).abs() < 1e-10);
    assert!((bfs.distance(3) - 3.0).abs() < 1e-10);
    assert!((bfs.distance(4) - 4.0).abs() < 1e-10);

    let all_dists = bfs.distances();
    assert_eq!(all_dists.len(), 5);
}

#[test]
fn test_bfs_disconnected() {
    let mut g = Graph::new(4, false, false);
    g.add_edge(0, 1, 1.0);
    // 2 and 3 are disconnected from 0

    let mut bfs = Bfs::new(&g, 0, false);
    bfs.run();

    assert!((bfs.distance(0) - 0.0).abs() < 1e-10);
    assert!((bfs.distance(1) - 1.0).abs() < 1e-10);
    // Disconnected nodes should have very large distance
    assert!(bfs.distance(2) > 1e15);
}
