/* =============================================================================
 * galileo/src/graph/galileo_graph.c - Graph Neural Network Module
 * 
 * Hot-loadable shared library implementing graph operations, message passing,
 * attention mechanisms, and dynamic edge management for Galileo v42.
 * 
 * UPDATED for full lazy loading support with proper module lifecycle management.
 * =============================================================================
 */

/* Feature test macros for maximum portability */
#if defined(__linux__) || defined(__GLIBC__)
    #define _GNU_SOURCE
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
    #define _POSIX_C_SOURCE 200112L
    #define _DEFAULT_SOURCE
#elif defined(__sun) || defined(__SVR4)
    #define _POSIX_C_SOURCE 200112L
    #define __EXTENSIONS__
#else
    #define _POSIX_C_SOURCE 200112L
#endif

#include "galileo_graph.h"
#include "../core/galileo_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/* =============================================================================
 * MODULE METADATA AND INITIALIZATION
 * =============================================================================
 */

static int graph_module_initialized = 0;

/* Module initialization function */
static int graph_module_init(void) {
    if (graph_module_initialized) {
        return 0;  /* Already initialized */
    }
    
    fprintf(stderr, "🔗 Graph module v42.1 initializing...\n");
    
    /* Any module-specific initialization would go here */
    /* For example: initialize thread pools, algorithm caches, etc. */
    
    graph_module_initialized = 1;
    fprintf(stderr, "✅ Graph module ready!\n");
    return 0;
}

/* Module cleanup function */
static void graph_module_cleanup(void) {
    if (!graph_module_initialized) {
        return;
    }
    
    fprintf(stderr, "🔗 Graph module shutting down...\n");
    
    /* Clean up any module-specific resources here */
    
    graph_module_initialized = 0;
    fprintf(stderr, "✅ Graph module cleaned up!\n");
}

/* Module info structure for dynamic loading */
GraphModuleInfo graph_module_info = {
    .name = "graph",
    .version = "42.1",
    .init_func = graph_module_init,
    .cleanup_func = graph_module_cleanup
};

/* =============================================================================
 * HASH TABLE UTILITIES FOR EDGE DEDUPLICATION
 * =============================================================================
 */

/* Hash function for edge deduplication - PHASE 0 enhancement */
uint32_t hash_edge_key(const EdgeKey* key) {
    uint32_t hash = 5381;
    hash = ((hash << 5) + hash) + key->src;
    hash = ((hash << 5) + hash) + key->dst;
    hash = ((hash << 5) + hash) + (uint32_t)key->type;
    return hash % EDGE_HASH_SIZE;
}

/* Check if edge already exists - PHASE 0 enhancement */
int edge_exists(GalileoModel* model, int src, int dst, EdgeType type) {
    if (!model || !graph_module_initialized) {
        return 0;
    }
    
    EdgeKey key = {src, dst, type};
    uint32_t hash = hash_edge_key(&key);
    
    EdgeHashEntry* entry = model->edge_hash[hash];
    while (entry) {
        if (entry->key.src == src && entry->key.dst == dst && entry->key.type == type) {
            return 1;  /* Found duplicate */
        }
        entry = entry->next;
    }
    return 0;  /* Not found */
}

/* Add edge to hash table - PHASE 0 enhancement */
void add_edge_to_hash(GalileoModel* model, int src, int dst, EdgeType type) {
    if (!model || !graph_module_initialized) {
        return;
    }
    
    EdgeKey key = {src, dst, type};
    uint32_t hash = hash_edge_key(&key);
    
    EdgeHashEntry* entry = malloc(sizeof(EdgeHashEntry));
    if (!entry) {
        fprintf(stderr, "⚠️  Failed to allocate memory for edge hash entry\n");
        return;
    }
    
    entry->key = key;
    entry->next = model->edge_hash[hash];
    model->edge_hash[hash] = entry;
}

/* =============================================================================
 * SIMILARITY AND ATTENTION COMPUTATION
 * =============================================================================
 */

/* Stable cosine similarity with NaN protection */
float stable_cosine_similarity(const float* a, const float* b, int dim) {
    if (!a || !b || dim <= 0 || !graph_module_initialized) {
        return 0.0f;
    }
    
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    
    for (int i = 0; i < dim; i++) {
        if (isfinite(a[i]) && isfinite(b[i])) {
            dot += (double)a[i] * (double)b[i];
            norm_a += (double)a[i] * (double)a[i];
            norm_b += (double)b[i] * (double)b[i];
        }
    }
    
    norm_a = sqrt(norm_a);
    norm_b = sqrt(norm_b);
    
    if (norm_a < 1e-10 || norm_b < 1e-10) {
        return 0.0f;
    }
    
    float similarity = (float)(dot / (norm_a * norm_b));
    return isfinite(similarity) ? similarity : 0.0f;
}

/* Multi-scale similarity computation with enhanced robustness */
float compute_multiscale_similarity_safe(const GraphNode* node1, const GraphNode* node2) {
    if (!node1 || !node2 || !graph_module_initialized) {
        return 0.0f;
    }
    
    /* Semantic similarity (embeddings) */
    float semantic_sim = stable_cosine_similarity(node1->embedding, node2->embedding, EMBEDDING_DIM);
    
    /* Token similarity (string matching) */
    float token_sim = (strcmp(node1->token, node2->token) == 0) ? 1.0f : 0.0f;
    
    /* Importance-weighted similarity */
    float importance_weight = sqrtf(node1->importance_score * node2->importance_score);
    importance_weight = isfinite(importance_weight) ? fminf(importance_weight, 1.0f) : 0.0f;
    
    /* Combine with weighted average */
    float combined = 0.6f * semantic_sim + 0.3f * token_sim + 0.1f * importance_weight;
    
    return isfinite(combined) ? fmaxf(0.0f, fminf(1.0f, combined)) : 0.0f;
}

/* Compute attention score between two nodes */
float compute_attention_score(GalileoModel* model, int src, int dst) {
    if (!model || !graph_module_initialized || src < 0 || dst < 0 || 
        src >= model->num_nodes || dst >= model->num_nodes || src == dst) {
        return 0.0f;
    }
    
    return compute_multiscale_similarity_safe(&model->nodes[src], &model->nodes[dst]);
}

/* Compute detailed attention with breakdown */
AttentionScore compute_detailed_attention(GalileoModel* model, int src, int dst) {
    AttentionScore score = {0};
    
    if (!model || !graph_module_initialized || src < 0 || dst < 0 || 
        src >= model->num_nodes || dst >= model->num_nodes || src == dst) {
        return score;
    }
    
    const GraphNode* node1 = &model->nodes[src];
    const GraphNode* node2 = &model->nodes[dst];
    
    score.semantic_similarity = stable_cosine_similarity(node1->embedding, node2->embedding, EMBEDDING_DIM);
    score.token_similarity = (strcmp(node1->token, node2->token) == 0) ? 1.0f : 0.0f;
    score.importance_factor = sqrtf(node1->importance_score * node2->importance_score);
    
    /* Ensure all components are finite */
    score.semantic_similarity = isfinite(score.semantic_similarity) ? score.semantic_similarity : 0.0f;
    score.importance_factor = isfinite(score.importance_factor) ? score.importance_factor : 0.0f;
    
    score.combined_score = 0.6f * score.semantic_similarity + 0.3f * score.token_similarity + 0.1f * score.importance_factor;
    score.combined_score = isfinite(score.combined_score) ? fmaxf(0.0f, fminf(1.0f, score.combined_score)) : 0.0f;
    
    return score;
}

/* =============================================================================
 * EDGE MANAGEMENT
 * =============================================================================
 */

/* Safe edge addition with deduplication checking */
int galileo_add_edge_safe(GalileoModel* model, int src, int dst, EdgeType type, float weight) {
    if (!model || !graph_module_initialized || src < 0 || dst < 0 || 
        src >= model->num_nodes || dst >= model->num_nodes || src == dst) {
        return -1;
    }
    
    /* Check for duplicates */
    if (edge_exists(model, src, dst, type)) {
        return 0;  /* Edge already exists, no need to add */
    }
    
    /* Check capacity */
    if (model->num_edges >= MAX_EDGES) {
        return -1;  /* No capacity for more edges */
    }
    
    /* Add the edge */
    GraphEdge* edge = &model->edges[model->num_edges];
    edge->src = src;
    edge->dst = dst;
    edge->type = type;
    edge->weight = isfinite(weight) ? weight : 0.0f;
    edge->activation = 0.0f;
    edge->last_updated = model->current_iteration;
    
    /* Add to hash table for deduplication */
    add_edge_to_hash(model, src, dst, type);
    
    model->num_edges++;
    return model->num_edges - 1;  /* Return edge index */
}

/* =============================================================================
 * MESSAGE PASSING AND GRAPH EVOLUTION
 * =============================================================================
 */

/* Core message passing iteration */
void galileo_message_passing_iteration(GalileoModel* model) {
    if (!model || !graph_module_initialized) {
        return;
    }
    
    /* Update edge activations based on attention */
    for (int i = 0; i < model->num_edges; i++) {
        GraphEdge* edge = &model->edges[i];
        if (edge->src >= 0 && edge->src < model->num_nodes && 
            edge->dst >= 0 && edge->dst < model->num_nodes) {
            
            float attention = compute_attention_score(model, edge->src, edge->dst);
            edge->activation = 0.9f * edge->activation + 0.1f * attention;
            edge->last_updated = model->current_iteration;
        }
    }
    
    /* Update node importance scores based on incoming messages */
    for (int i = 0; i < model->num_nodes; i++) {
        float total_incoming = 0.0f;
        int incoming_count = 0;
        
        for (int j = 0; j < model->num_edges; j++) {
            if (model->edges[j].dst == i) {
                total_incoming += model->edges[j].activation * model->edges[j].weight;
                incoming_count++;
            }
        }
        
        if (incoming_count > 0) {
            float avg_incoming = total_incoming / incoming_count;
            model->nodes[i].importance_score = 0.8f * model->nodes[i].importance_score + 0.2f * avg_incoming;
            
            /* Ensure importance score stays in valid range */
            model->nodes[i].importance_score = fmaxf(0.0f, fminf(1.0f, model->nodes[i].importance_score));
        }
    }
}

/* Comparison function for sorting edge candidates by score (descending) */
int compare_edge_candidates(const void* a, const void* b) {
    const EdgeCandidate* cand_a = (const EdgeCandidate*)a;
    const EdgeCandidate* cand_b = (const EdgeCandidate*)b;
    
    if (cand_a->score > cand_b->score) return -1;
    if (cand_a->score < cand_b->score) return 1;
    return 0;
}

/* Smart edge candidate generation with O(n log n) optimization */
void galileo_smart_edge_candidates(GalileoModel* model) {
    if (!model || !graph_module_initialized) {
        return;
    }
    
    EdgeCandidate candidates[MAX_EDGE_CANDIDATES];
    int candidate_count = 0;
    
    /* Generate candidates by sampling high-importance nodes */
    for (int i = 0; i < model->num_nodes && candidate_count < MAX_EDGE_CANDIDATES; i++) {
        if (model->nodes[i].importance_score < 0.3f) continue;  /* Skip low-importance nodes */
        
        for (int j = i + 1; j < model->num_nodes && candidate_count < MAX_EDGE_CANDIDATES; j++) {
            if (model->nodes[j].importance_score < 0.3f) continue;
            
            if (!edge_exists(model, i, j, EDGE_SEMANTIC)) {
                float score = compute_attention_score(model, i, j);
                if (score > model->similarity_threshold) {
                    candidates[candidate_count].src = i;
                    candidates[candidate_count].dst = j;
                    candidates[candidate_count].score = score;
                    candidate_count++;
                }
            }
        }
    }
    
    /* Sort candidates by score and add the best ones */
    qsort(candidates, candidate_count, sizeof(EdgeCandidate), compare_edge_candidates);
    
    int added_edges = 0;
    for (int i = 0; i < candidate_count && added_edges < 10; i++) {
        int result = galileo_add_edge_safe(model, candidates[i].src, candidates[i].dst, 
                                          EDGE_SEMANTIC, candidates[i].score);
        if (result >= 0) {
            added_edges++;
        }
    }
    
    if (added_edges > 0) {
        printf("🔗 Added %d new semantic edges from %d candidates\n", added_edges, candidate_count);
    }
}

/* Attention-based edge addition with optimization */
void galileo_attention_based_edge_addition_optimized(GalileoModel* model) {
    if (!model || !graph_module_initialized) {
        return;
    }
    
    galileo_smart_edge_candidates(model);
}

/* Prune weak edges to maintain graph sparsity */
void galileo_prune_weak_edges(GalileoModel* model) {
    if (!model || !graph_module_initialized) {
        return;
    }
    
    int removed_count = 0;
    
    /* Mark weak edges for removal (iterate backwards to avoid index shifting) */
    for (int i = model->num_edges - 1; i >= 0; i--) {
        GraphEdge* edge = &model->edges[i];
        
        /* Remove edges that are very weak and haven't been updated recently */
        if (edge->activation < 0.1f && 
            (model->current_iteration - edge->last_updated) > 5) {
            
            /* Remove from hash table by rebuilding it */
            /* For now, we'll just mark the edge as invalid */
            edge->weight = 0.0f;
            edge->activation = 0.0f;
            
            /* Compact the edge array */
            memmove(&model->edges[i], &model->edges[i + 1], 
                   (model->num_edges - i - 1) * sizeof(GraphEdge));
            model->num_edges--;
            removed_count++;
        }
    }
    
    if (removed_count > 0) {
        printf("🧹 Pruned %d weak edges\n", removed_count);
    }
}
