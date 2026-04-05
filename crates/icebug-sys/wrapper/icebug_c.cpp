#include "icebug_c.h"

#include <networkit/graph/Graph.hpp>
#include <networkit/graph/GraphW.hpp>
#include <networkit/centrality/PageRank.hpp>
#include <networkit/community/PLM.hpp>
#include <networkit/structures/Partition.hpp>
#include <networkit/distance/BFS.hpp>

#include <vector>
#include <cstring>

// ---- Opaque wrappers ----
struct IcebugGraph {
    NetworKit::GraphW inner;
    IcebugGraph(uint64_t n, bool weighted, bool directed)
        : inner(n, weighted, directed) {}
};

struct IcebugPageRank {
    NetworKit::PageRank inner;
    IcebugPageRank(const NetworKit::Graph& g, double damping, double tol)
        : inner(g, damping, tol) {}
};

struct IcebugLouvain {
    NetworKit::PLM inner;
    IcebugLouvain(const NetworKit::Graph& g, bool refine, double gamma, uint64_t maxIter)
        : inner(g, refine, gamma, "balanced", maxIter) {}
};

struct IcebugPartition {
    NetworKit::Partition inner;
    IcebugPartition(NetworKit::Partition p) : inner(std::move(p)) {}
};

struct IcebugBFS {
    NetworKit::BFS inner;
    IcebugBFS(const NetworKit::Graph& g, uint64_t source, bool storePaths)
        : inner(g, source, storePaths) {}
};

// ---- Graph lifecycle ----

extern "C" IcebugGraph* icebug_graph_new(uint64_t n, bool weighted, bool directed) {
    return new IcebugGraph(n, weighted, directed);
}

extern "C" void icebug_graph_free(IcebugGraph* g) {
    delete g;
}

// ---- Graph mutation ----

extern "C" uint64_t icebug_graph_add_node(IcebugGraph* g) {
    return g->inner.addNode();
}

extern "C" void icebug_graph_add_nodes(IcebugGraph* g, uint64_t count) {
    g->inner.addNodes(count);
}

extern "C" void icebug_graph_add_edge(IcebugGraph* g, uint64_t u, uint64_t v, double weight) {
    g->inner.addEdge(u, v, weight);
}

extern "C" void icebug_graph_remove_edge(IcebugGraph* g, uint64_t u, uint64_t v) {
    g->inner.removeEdge(u, v);
}

extern "C" void icebug_graph_remove_node(IcebugGraph* g, uint64_t u) {
    g->inner.removeNode(u);
}

extern "C" void icebug_graph_set_weight(IcebugGraph* g, uint64_t u, uint64_t v, double w) {
    g->inner.setWeight(u, v, w);
}

// ---- Graph queries ----

extern "C" uint64_t icebug_graph_number_of_nodes(const IcebugGraph* g) {
    return g->inner.numberOfNodes();
}

extern "C" uint64_t icebug_graph_number_of_edges(const IcebugGraph* g) {
    return g->inner.numberOfEdges();
}

extern "C" bool icebug_graph_has_edge(const IcebugGraph* g, uint64_t u, uint64_t v) {
    return g->inner.hasEdge(u, v);
}

extern "C" double icebug_graph_weight(const IcebugGraph* g, uint64_t u, uint64_t v) {
    return g->inner.weight(u, v);
}

extern "C" uint64_t icebug_graph_degree(const IcebugGraph* g, uint64_t u) {
    return g->inner.degree(u);
}

extern "C" uint64_t icebug_graph_degree_in(const IcebugGraph* g, uint64_t u) {
    return g->inner.degreeIn(u);
}

extern "C" uint64_t icebug_graph_degree_out(const IcebugGraph* g, uint64_t u) {
    return g->inner.degreeOut(u);
}

extern "C" bool icebug_graph_is_weighted(const IcebugGraph* g) {
    return g->inner.isWeighted();
}

extern "C" bool icebug_graph_is_directed(const IcebugGraph* g) {
    return g->inner.isDirected();
}

extern "C" bool icebug_graph_has_node(const IcebugGraph* g, uint64_t u) {
    return g->inner.hasNode(u);
}

extern "C" uint64_t icebug_graph_upper_node_id_bound(const IcebugGraph* g) {
    return g->inner.upperNodeIdBound();
}

// ---- Neighbor access ----

extern "C" uint64_t icebug_graph_neighbors(const IcebugGraph* g, uint64_t u, uint64_t* out, uint64_t out_len) {
    uint64_t count = 0;
    g->inner.forNeighborsOf(u, [&](uint64_t v) {
        if (out && count < out_len) {
            out[count] = v;
        }
        count++;
    });
    return count;
}

extern "C" uint64_t icebug_graph_in_neighbors(const IcebugGraph* g, uint64_t u, uint64_t* out, uint64_t out_len) {
    uint64_t count = 0;
    g->inner.forInNeighborsOf(u, [&](uint64_t v) {
        if (out && count < out_len) {
            out[count] = v;
        }
        count++;
    });
    return count;
}

// ---- Iteration helpers ----

extern "C" void icebug_graph_for_edges(const IcebugGraph* g, IcebugEdgeCallback callback, void* user_data) {
    g->inner.forEdges([&](uint64_t u, uint64_t v, double w, uint64_t eid) {
        callback(user_data, u, v, w, eid);
    });
}

extern "C" void icebug_graph_for_nodes(const IcebugGraph* g, IcebugNodeCallback callback, void* user_data) {
    g->inner.forNodes([&](uint64_t u) {
        callback(user_data, u);
    });
}

// ---- PageRank ----

extern "C" IcebugPageRank* icebug_pagerank_new(const IcebugGraph* g, double damping, double tolerance) {
    return new IcebugPageRank(g->inner, damping, tolerance);
}

extern "C" void icebug_pagerank_run(IcebugPageRank* pr) {
    pr->inner.run();
}

extern "C" double icebug_pagerank_score(IcebugPageRank* pr, uint64_t node) {
    return pr->inner.score(node);
}

extern "C" uint64_t icebug_pagerank_scores(IcebugPageRank* pr, double* out, uint64_t out_len) {
    const auto& scores = pr->inner.scores();
    uint64_t n = std::min((uint64_t)scores.size(), out_len);
    std::memcpy(out, scores.data(), n * sizeof(double));
    return n;
}

extern "C" void icebug_pagerank_free(IcebugPageRank* pr) {
    delete pr;
}

// ---- Louvain (PLM) ----

extern "C" IcebugLouvain* icebug_louvain_new(const IcebugGraph* g, bool refine, double gamma, uint64_t max_iter) {
    return new IcebugLouvain(g->inner, refine, gamma, max_iter);
}

extern "C" void icebug_louvain_run(IcebugLouvain* lv) {
    lv->inner.run();
}

extern "C" IcebugPartition* icebug_louvain_get_partition(const IcebugLouvain* lv) {
    return new IcebugPartition(lv->inner.getPartition());
}

extern "C" void icebug_louvain_free(IcebugLouvain* lv) {
    delete lv;
}

// ---- Partition ----

extern "C" uint64_t icebug_partition_subset_of(const IcebugPartition* p, uint64_t node) {
    return p->inner.subsetOf(node);
}

extern "C" uint64_t icebug_partition_number_of_subsets(const IcebugPartition* p) {
    return p->inner.numberOfSubsets();
}

extern "C" uint64_t icebug_partition_get_vector(const IcebugPartition* p, uint64_t* out, uint64_t out_len) {
    const auto& vec = p->inner.getVector();
    uint64_t n = std::min((uint64_t)vec.size(), out_len);
    std::memcpy(out, vec.data(), n * sizeof(uint64_t));
    return n;
}

extern "C" void icebug_partition_free(IcebugPartition* p) {
    delete p;
}

// ---- BFS ----

extern "C" IcebugBFS* icebug_bfs_new(const IcebugGraph* g, uint64_t source, bool store_paths) {
    return new IcebugBFS(g->inner, source, store_paths);
}

extern "C" void icebug_bfs_run(IcebugBFS* bfs) {
    bfs->inner.run();
}

extern "C" double icebug_bfs_distance(IcebugBFS* bfs, uint64_t target) {
    return bfs->inner.distance(target);
}

extern "C" uint64_t icebug_bfs_distances(IcebugBFS* bfs, double* out, uint64_t out_len) {
    const auto& dists = bfs->inner.getDistances();
    uint64_t n = std::min((uint64_t)dists.size(), out_len);
    std::memcpy(out, dists.data(), n * sizeof(double));
    return n;
}

extern "C" void icebug_bfs_free(IcebugBFS* bfs) {
    delete bfs;
}
