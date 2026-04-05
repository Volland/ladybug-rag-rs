#ifndef ICEBUG_C_H
#define ICEBUG_C_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handles
typedef struct IcebugGraph IcebugGraph;
typedef struct IcebugPageRank IcebugPageRank;
typedef struct IcebugLouvain IcebugLouvain;
typedef struct IcebugPartition IcebugPartition;
typedef struct IcebugBFS IcebugBFS;

// --- Graph lifecycle ---
IcebugGraph* icebug_graph_new(uint64_t n, bool weighted, bool directed);
void icebug_graph_free(IcebugGraph* g);

// --- Graph mutation ---
uint64_t icebug_graph_add_node(IcebugGraph* g);
void icebug_graph_add_nodes(IcebugGraph* g, uint64_t count);
void icebug_graph_add_edge(IcebugGraph* g, uint64_t u, uint64_t v, double weight);
void icebug_graph_remove_edge(IcebugGraph* g, uint64_t u, uint64_t v);
void icebug_graph_remove_node(IcebugGraph* g, uint64_t u);
void icebug_graph_set_weight(IcebugGraph* g, uint64_t u, uint64_t v, double w);

// --- Graph queries ---
uint64_t icebug_graph_number_of_nodes(const IcebugGraph* g);
uint64_t icebug_graph_number_of_edges(const IcebugGraph* g);
bool icebug_graph_has_edge(const IcebugGraph* g, uint64_t u, uint64_t v);
double icebug_graph_weight(const IcebugGraph* g, uint64_t u, uint64_t v);
uint64_t icebug_graph_degree(const IcebugGraph* g, uint64_t u);
uint64_t icebug_graph_degree_in(const IcebugGraph* g, uint64_t u);
uint64_t icebug_graph_degree_out(const IcebugGraph* g, uint64_t u);
bool icebug_graph_is_weighted(const IcebugGraph* g);
bool icebug_graph_is_directed(const IcebugGraph* g);
bool icebug_graph_has_node(const IcebugGraph* g, uint64_t u);
uint64_t icebug_graph_upper_node_id_bound(const IcebugGraph* g);

// --- Neighbor access ---
// Writes neighbor node IDs into `out` buffer. Returns number of neighbors written.
// If `out` is NULL, just returns the count.
uint64_t icebug_graph_neighbors(const IcebugGraph* g, uint64_t u, uint64_t* out, uint64_t out_len);
uint64_t icebug_graph_in_neighbors(const IcebugGraph* g, uint64_t u, uint64_t* out, uint64_t out_len);

// --- Iteration helpers ---
// Calls `callback(user_data, u, v, weight, edge_id)` for each edge
typedef void (*IcebugEdgeCallback)(void* user_data, uint64_t u, uint64_t v, double weight, uint64_t edge_id);
void icebug_graph_for_edges(const IcebugGraph* g, IcebugEdgeCallback callback, void* user_data);

typedef void (*IcebugNodeCallback)(void* user_data, uint64_t node);
void icebug_graph_for_nodes(const IcebugGraph* g, IcebugNodeCallback callback, void* user_data);

// --- PageRank ---
IcebugPageRank* icebug_pagerank_new(const IcebugGraph* g, double damping, double tolerance);
void icebug_pagerank_run(IcebugPageRank* pr);
double icebug_pagerank_score(IcebugPageRank* pr, uint64_t node);
// Writes all scores into `out`. Returns number of scores written.
uint64_t icebug_pagerank_scores(IcebugPageRank* pr, double* out, uint64_t out_len);
void icebug_pagerank_free(IcebugPageRank* pr);

// --- Louvain (PLM) Community Detection ---
IcebugLouvain* icebug_louvain_new(const IcebugGraph* g, bool refine, double gamma, uint64_t max_iter);
void icebug_louvain_run(IcebugLouvain* lv);
// Returns a Partition handle (caller must free)
IcebugPartition* icebug_louvain_get_partition(const IcebugLouvain* lv);
void icebug_louvain_free(IcebugLouvain* lv);

// --- Partition ---
uint64_t icebug_partition_subset_of(const IcebugPartition* p, uint64_t node);
uint64_t icebug_partition_number_of_subsets(const IcebugPartition* p);
// Writes subset IDs for all nodes into `out`. Returns count.
uint64_t icebug_partition_get_vector(const IcebugPartition* p, uint64_t* out, uint64_t out_len);
void icebug_partition_free(IcebugPartition* p);

// --- BFS ---
IcebugBFS* icebug_bfs_new(const IcebugGraph* g, uint64_t source, bool store_paths);
void icebug_bfs_run(IcebugBFS* bfs);
double icebug_bfs_distance(IcebugBFS* bfs, uint64_t target);
// Writes distances for all nodes into `out`. Returns count.
uint64_t icebug_bfs_distances(IcebugBFS* bfs, double* out, uint64_t out_len);
void icebug_bfs_free(IcebugBFS* bfs);

#ifdef __cplusplus
}
#endif

#endif // ICEBUG_C_H
