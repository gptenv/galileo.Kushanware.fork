/* =============================================================================
 * galileo/src/graph/galileo_graph.c - Graph Neural Network Module
 * 
 * Hot-loadable shared library implementing graph operations, message passing,
 * attention mechanisms, and dynamic edge management for Galileo v42.
 * 
 * Extracted from galileo_legacy_core-v42-v3.pre-modular.best.c and optimized
 * for modular architecture with O(n²) → O(n log n) improvements.
 * =============================================================================
 */

#include "galileo_graph.h"
#include "../core/galileo_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* =============================================================================
 * MODULE METADATA AND INITIALIZATION
 * =============================================================================
 */

static int graph_module_initialized = 0;

/* Module initialization */
static int graph_module_init(void) {
    if (graph_module_initialized) {
        return 0;  /* Already initialized */
    }
    
    fprintf(stderr, "🔗 Graph module v42.1 initializing...\n");
    
    /* Any module-specific initialization would go here */
    
    graph_module_initialized = 1;
    fprintf(stderr, "✅ Graph module ready!\n");
    return 0;
}

/* Module cleanup */
static void graph_module_cleanup(void) {
    if (!graph_module_initialized) {
        return;
    }
    
    fprintf(stderr, "🔗 Graph module shutting down...\n");
    graph_module_initialized = 0;
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
    EdgeKey key = {src, dst, type};
    uint32_t hash = hash_edge_key(&key);
    
    EdgeHashEntry* entry = malloc(sizeof(EdgeHashEntry));
    entry->key = key;
    entry->next = model->edge_hash[hash];
    model->edge_hash[hash] = entry;
}

/* =============================================================================
 * ENHANCED EDGE MANAGEMENT WITH DEDUPLICATION
 * =============================================================================
 */

/* Safe edge addition with duplicate checking - PHASE 0 enhancement */
int galileo_add_edge_safe(GalileoModel* model, int src, int dst, EdgeType type, float weight) {
    if (!model) return -1;
    if (src < 0 || src >= model->num_nodes || dst < 0 || dst >= model->num_nodes) {
        return -1;  /* Invalid node indices */
    }
    if (model->num_edges >= MAX_EDGES) {
        return -1;  /* Edge limit reached */
    }
    
    /* Check for duplicate */
    if (edge_exists(model, src, dst, type)) {
        return 0;  /* Edge already exists, no error but no action */
    }
    
    /* Add the edge */
    GraphEdge* edge = &model->edges[model->num_edges];
    edge->src = src;
    edge->dst = dst;
    edge->type = type;
    edge->weight = weight;
    edge->active = 1;
    edge->attention_score = 0.0f;
    edge->last_updated = model->current_iteration;
    
    /* Add to hash table for future deduplication */
    add_edge_to_hash(model, src, dst, type);
    
    model->num_edges++;
    model->total_edges_added++;
    
    return 1;  /* Successfully added */
}

/* =============================================================================
 * ENHANCED SIMILARITY AND ATTENTION COMPUTATION
 * =============================================================================
 */

/* Stable cosine similarity with NaN protection */
float stable_cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);
    
    /* Prevent division by zero and NaN */
    if (norm_a < 1e-8f || norm_b < 1e-8f) {
        return 0.0f;
    }
    
    float sim = dot / (norm_a * norm_b);
    
    /* Clamp to [-1, 1] to handle floating point errors */
    if (sim > 1.0f) sim = 1.0f;
    if (sim < -1.0f) sim = -1.0f;
    
    return sim;
}

/* Multi-scale similarity with NaN protection - PHASE 0 enhancement */
float compute_multiscale_similarity_safe(const GraphNode* node1, const GraphNode* node2) {
    if (!node1 || !node2 || !node1->active || !node2->active) {
        return 0.0f;
    }
    
    /* Identity similarity */
    float identity_sim = stable_cosine_similarity(node1->identity_embedding, 
                                                 node2->identity_embedding, EMBEDDING_DIM);
    
    /* Context similarity */
    float context_sim = stable_cosine_similarity(node1->context_embedding, 
                                                node2->context_embedding, EMBEDDING_DIM);
    
    /* Temporal similarity */
    float temporal_sim = stable_cosine_similarity(node1->temporal_embedding, 
                                                 node2->temporal_embedding, EMBEDDING_DIM);
    
    /* Weighted combination */
    float combined_sim = 0.5f * identity_sim + 0.3f * context_sim + 0.2f * temporal_sim;
    
    /* Adaptive importance-based weighting with NaN protection */
    float avg_importance = (node1->importance_score + node2->importance_score) / 2.0f;
    if (avg_importance < 1e-6f) avg_importance = 1e-6f;  /* Prevent underflow */
    
    float exponent = 1.5f + 0.5f * avg_importance;
    if (exponent > 10.0f) exponent = 10.0f;  /* Prevent overflow */
    
    /* Ensure positive base for power function */
    float base = fmaxf(fabsf(combined_sim), 1e-6f);
    float enhanced_sim = powf(base, exponent);
    
    /* Preserve sign and clamp result */
    if (combined_sim < 0.0f) enhanced_sim = -enhanced_sim;
    
    return fmaxf(-1.0f, fminf(1.0f, enhanced_sim));
}

/* Enhanced attention score computation */
float compute_attention_score(GalileoModel* model, int src, int dst) {
    if (!model || src < 0 || src >= model->num_nodes || dst < 0 || dst >= model->num_nodes) {
        return 0.0f;
    }
    
    GraphNode* src_node = &model->nodes[src];
    GraphNode* dst_node = &model->nodes[dst];
    
    if (!src_node->active || !dst_node->active) {
        return 0.0f;
    }
    
    /* Base similarity score */
    float base_score = compute_multiscale_similarity_safe(src_node, dst_node);
    
    /* Importance boost */
    float importance_factor = (src_node->importance_score + dst_node->importance_score) / 2.0f;
    
    /* Temporal recency boost */
    int src_recency = model->current_iteration - src_node->last_accessed_iteration;
    int dst_recency = model->current_iteration - dst_node->last_accessed_iteration;
    float recency_factor = 1.0f / (1.0f + (src_recency + dst_recency) / 10.0f);
    
    /* Combined attention score */
    float attention = base_score * (1.0f + 0.5f * importance_factor) * (0.5f + 0.5f * recency_factor);
    
    return fmaxf(0.0f, fminf(1.0f, attention));
}

/* Detailed attention computation with metadata */
AttentionScore compute_detailed_attention(GalileoModel* model, int src, int dst) {
    AttentionScore result = {0};
    
    if (!model || src < 0 || src >= model->num_nodes || dst < 0 || dst >= model->num_nodes) {
        strcpy(result.reason, "Invalid node indices");
        return result;
    }
    
    GraphNode* src_node = &model->nodes[src];
    GraphNode* dst_node = &model->nodes[dst];
    
    if (!src_node->active || !dst_node->active) {
        strcpy(result.reason, "Inactive nodes");
        return result;
    }
    
    /* Compute detailed similarity components */
    result.identity_similarity = stable_cosine_similarity(src_node->identity_embedding, 
                                                         dst_node->identity_embedding, EMBEDDING_DIM);
    result.context_similarity = stable_cosine_similarity(src_node->context_embedding, 
                                                        dst_node->context_embedding, EMBEDDING_DIM);
    result.temporal_similarity = stable_cosine_similarity(src_node->temporal_embedding, 
                                                         dst_node->temporal_embedding, EMBEDDING_DIM);
    
    /* Overall attention score */
    result.attention_score = compute_attention_score(model, src, dst);
    
    /* Generate explanation */
    snprintf(result.reason, sizeof(result.reason) - 1, 
             "ID:%.2f CTX:%.2f TMP:%.2f → ATT:%.2f", 
             result.identity_similarity, result.context_similarity, 
             result.temporal_similarity, result.attention_score);
    result.reason[sizeof(result.reason) - 1] = '\0';
    
    return result;
}

/* =============================================================================
 * MESSAGE PASSING ITERATIONS
 * =============================================================================
 */

/* Enhanced message passing with multi-scale updates */
void galileo_message_passing_iteration(GalileoModel* model) {
    if (!model || model->num_nodes == 0) return;
    
    /* Clear message buffers */
    memset(model->node_messages_local, 0, sizeof(model->node_messages_local));
    memset(model->node_messages_global, 0, sizeof(model->node_messages_global));
    memset(model->node_messages_attention, 0, sizeof(model->node_messages_attention));
    memset(model->node_updates, 0, sizeof(model->node_updates));
    
    /* Phase 1: Collect messages from all edges */
    for (int e = 0; e < model->num_edges; e++) {
        GraphEdge* edge = &model->edges[e];
        if (!edge->active) continue;
        
        GraphNode* src = &model->nodes[edge->src];
        GraphNode* dst = &model->nodes[edge->dst];
        
        if (!src->active || !dst->active) continue;
        
        /* Compute message strength based on edge type */
        float message_strength = edge->weight * edge->attention_score;
        
        /* Local message passing (identity embeddings) */
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            model->node_messages_local[edge->dst][d] += 
                src->identity_embedding[d] * message_strength;
        }
        
        /* Context-aware message passing */
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            model->node_messages_global[edge->dst][d] += 
                src->context_embedding[d] * message_strength * 0.7f;
        }
        
        /* Attention-based message passing */
        float attention = compute_attention_score(model, edge->src, edge->dst);
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            model->node_messages_attention[edge->dst][d] += 
                src->temporal_embedding[d] * attention;
        }
    }
    
    /* Phase 2: Update node embeddings with gated combination */
    for (int n = 0; n < model->num_nodes; n++) {
        GraphNode* node = &model->nodes[n];
        if (!node->active) continue;
        
        /* Compute gating factors based on node importance */
        float local_gate = 0.6f + 0.2f * node->importance_score;
        float global_gate = 0.3f + 0.1f * node->importance_score;
        float attention_gate = 0.1f + 0.2f * node->importance_score;
        
        /* Normalize gates */
        float gate_sum = local_gate + global_gate + attention_gate;
        local_gate /= gate_sum;
        global_gate /= gate_sum;
        attention_gate /= gate_sum;
        
        /* Update each embedding type */
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            /* Combined update vector */
            model->node_updates[n][d] = 
                local_gate * model->node_messages_local[n][d] +
                global_gate * model->node_messages_global[n][d] +
                attention_gate * model->node_messages_attention[n][d];
            
            /* Apply updates with momentum and decay */
            float learning_rate = 0.1f;
            float momentum = 0.9f;
            
            /* Identity embedding update */
            node->identity_embedding[d] = momentum * node->identity_embedding[d] + 
                                         learning_rate * model->node_updates[n][d];
            
            /* Context embedding update */
            node->context_embedding[d] = momentum * node->context_embedding[d] + 
                                        learning_rate * model->node_messages_global[n][d];
            
            /* Temporal embedding update */
            node->temporal_embedding[d] = momentum * node->temporal_embedding[d] + 
                                         learning_rate * model->node_messages_attention[n][d];
        }
        
        /* Update node metadata */
        node->last_accessed_iteration = model->current_iteration;
    }
}

/* =============================================================================
 * DYNAMIC EDGE ADDITION AND OPTIMIZATION
 * =============================================================================
 */

/* Comparison function for qsort - PHASE 1 optimization */
int compare_edge_candidates(const void* a, const void* b) {
    const EdgeCandidate* ca = (const EdgeCandidate*)a;
    const EdgeCandidate* cb = (const EdgeCandidate*)b;
    
    /* Sort by attention score in descending order */
    if (ca->attention_score > cb->attention_score) return -1;
    if (ca->attention_score < cb->attention_score) return 1;
    return 0;
}

/* Optimized attention-based edge addition - PHASE 1 enhancement */
void galileo_attention_based_edge_addition_optimized(GalileoModel* model) {
    if (!model || model->num_nodes < 2) return;
    
    model->num_candidates = 0;
    
    /* Generate candidate edges with O(n²) but early termination */
    for (int i = 0; i < model->num_nodes && model->num_candidates < 1000; i++) {
        if (!model->nodes[i].active) continue;
        
        for (int j = i + 1; j < model->num_nodes && model->num_candidates < 1000; j++) {
            if (!model->nodes[j].active) continue;
            
            /* Skip if edge already exists */
            if (edge_exists(model, i, j, EDGE_SIMILARITY)) continue;
            
            /* Compute attention score */
            float attention = compute_attention_score(model, i, j);
            
            /* Only consider high-attention pairs */
            if (attention > model->attention_threshold) {
                EdgeCandidate* candidate = &model->edge_candidates[model->num_candidates];
                candidate->src = i;
                candidate->dst = j;
                candidate->type = EDGE_SIMILARITY;
                candidate->attention_score = attention;
                
                snprintf(candidate->reason, sizeof(candidate->reason) - 1,
                         "High attention: %.3f (nodes %d->%d)", attention, i, j);
                candidate->reason[sizeof(candidate->reason) - 1] = '\0';
                
                model->num_candidates++;
            }
        }
    }
    
    /* Sort candidates by attention score using qsort - PHASE 1 optimization */
    if (model->num_candidates > 0) {
        qsort(model->edge_candidates, model->num_candidates, 
              sizeof(EdgeCandidate), compare_edge_candidates);
    }
    
    /* Add top candidates as edges */
    int edges_to_add = model->max_edges_per_iteration;
    if (edges_to_add > model->num_candidates) {
        edges_to_add = model->num_candidates;
    }
    
    for (int i = 0; i < edges_to_add; i++) {
        EdgeCandidate* candidate = &model->edge_candidates[i];
        
        int success = galileo_add_edge_safe(model, candidate->src, candidate->dst, 
                                           candidate->type, candidate->attention_score);
        
        if (success > 0) {
            printf("🔗 Added edge %d->%d (attention: %.3f): %s\n", 
                   candidate->src, candidate->dst, candidate->attention_score, candidate->reason);
        }
    }
}

/* Smart edge candidate generation with heuristics */
void galileo_smart_edge_candidates(GalileoModel* model) {
    if (!model) return;
    
    /* This is a placeholder for more sophisticated candidate generation */
    /* Could include: semantic clustering, structural analysis, etc. */
    
    /* For now, delegate to the optimized attention-based method */
    galileo_attention_based_edge_addition_optimized(model);
}

/* Prune weak edges to maintain graph sparsity */
void galileo_prune_weak_edges(GalileoModel* model) {
    if (!model) return;
    
    int pruned_count = 0;
    
    for (int e = 0; e < model->num_edges; e++) {
        GraphEdge* edge = &model->edges[e];
        if (!edge->active) continue;
        
        /* Check if edge is weak */
        int staleness = model->current_iteration - edge->last_updated;
        if (edge->weight < 0.1f && staleness > 5) {
            edge->active = 0;
            pruned_count++;
        }
    }
    
    if (pruned_count > 0) {
        printf("✂️  Pruned %d weak edges\n", pruned_count);
    }
}
