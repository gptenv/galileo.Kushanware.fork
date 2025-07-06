/* =============================================================================
 * galileo/src/graph/galileo_graph.h - Graph Module Public API
 * 
 * Public header for the graph neural network module containing all functions
 * for graph operations, message passing, attention computation, and dynamic
 * edge management.
 * 
 * This module handles the core graph neural network functionality that makes
 * Galileo's dynamic graph architecture possible.
 * 
 * UPDATED for full lazy loading support with proper module lifecycle management.
 * =============================================================================
 */

#ifndef GALILEO_GRAPH_H
#define GALILEO_GRAPH_H

#include "../core/galileo_types.h"
#include <stdint.h>

/* =============================================================================
 * EDGE MANAGEMENT FUNCTIONS
 * =============================================================================
 */

/* Safe edge addition with deduplication checking */
int galileo_add_edge_safe(GalileoModel* model, int src, int dst, EdgeType type, float weight);

/* Edge existence and hash table management */
int edge_exists(GalileoModel* model, int src, int dst, EdgeType type);
void add_edge_to_hash(GalileoModel* model, int src, int dst, EdgeType type);
uint32_t hash_edge_key(const EdgeKey* key);

/* =============================================================================
 * SIMILARITY AND ATTENTION COMPUTATION
 * =============================================================================
 */

/* Multi-scale similarity computation with NaN protection */
float compute_multiscale_similarity_safe(const GraphNode* node1, const GraphNode* node2);

/* Attention score computation */
float compute_attention_score(GalileoModel* model, int src, int dst);
AttentionScore compute_detailed_attention(GalileoModel* model, int src, int dst);

/* Low-level similarity utilities */
float stable_cosine_similarity(const float* a, const float* b, int dim);

/* =============================================================================
 * MESSAGE PASSING AND GRAPH EVOLUTION
 * =============================================================================
 */

/* Core message passing iteration */
void galileo_message_passing_iteration(GalileoModel* model);

/* Dynamic edge addition and optimization */
void galileo_attention_based_edge_addition_optimized(GalileoModel* model);
void galileo_smart_edge_candidates(GalileoModel* model);
void galileo_prune_weak_edges(GalileoModel* model);

/* =============================================================================
 * UTILITY FUNCTIONS
 * =============================================================================
 */

/* Comparison function for sorting edge candidates */
int compare_edge_candidates(const void* a, const void* b);

/* =============================================================================
 * MODULE INTERFACE FOR LAZY LOADING
 * =============================================================================
 */

/* Module info structure for dynamic loading */
typedef struct {
    const char* name;
    const char* version;
    int (*init_func)(void);
    void (*cleanup_func)(void);
} GraphModuleInfo;

/* Module info instance - exported for module loader */
extern GraphModuleInfo graph_module_info;

/* =============================================================================
 * MODULE CAPABILITY CHECKS
 * =============================================================================
 */

/* Check if graph module is properly initialized */
static inline int galileo_graph_module_ready(void) {
    /* This would be implemented by checking initialization state */
    return 1; /* For now, assume ready if header is included */
}

#endif /* GALILEO_GRAPH_H */
