/* =============================================================================
 * galileo/src/memory/galileo_memory.c - Memory Management & Compression Module
 * 
 * Hot-loadable shared library implementing contextual memory addressing,
 * adaptive compression, hierarchical summarization, and intelligent forgetting.
 * 
 * This module gives Galileo the ability to compress information hierarchically,
 * retrieve memories based on context, and manage memory slots with importance-
 * based eviction. It's what makes Galileo scale beyond just keeping everything.
 * 
 * Extracted from galileo_legacy_core-v42-v3.pre-modular.best.c with enhanced
 * context-aware retrieval, smarter compression heuristics, and LRU management.
 * =============================================================================
 */

#include "galileo_memory.h"
#include "../core/galileo_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>

/* =============================================================================
 * MODULE METADATA AND INITIALIZATION
 * =============================================================================
 */

static int memory_module_initialized = 0;

/* Module initialization */
static int memory_module_init(void) {
    if (memory_module_initialized) {
        return 0;  /* Already initialized */
    }
    
    fprintf(stderr, "💾 Memory module v42.1 initializing...\n");
    
    /* Any module-specific initialization would go here */
    
    memory_module_initialized = 1;
    fprintf(stderr, "✅ Memory module ready! Contextual addressing & compression online.\n");
    return 0;
}

/* Module cleanup */
static void memory_module_cleanup(void) {
    if (!memory_module_initialized) {
        return;
    }
    
    fprintf(stderr, "💾 Memory module shutting down...\n");
    memory_module_initialized = 0;
}

/* Module info structure for dynamic loading */
MemoryModuleInfo memory_module_info = {
    .name = "memory",
    .version = "42.1",
    .init_func = memory_module_init,
    .cleanup_func = memory_module_cleanup
};

/* =============================================================================
 * UTILITY FUNCTIONS FOR MEMORY OPERATIONS
 * =============================================================================
 */

/* Compute cosine similarity for memory retrieval */
static float cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);
    
    if (norm_a < 1e-8f || norm_b < 1e-8f) {
        return 0.0f;
    }
    
    return dot / (norm_a * norm_b);
}

/* Generate contextual key by combining query and context */
static void generate_contextual_key(const float* query, const float* context, float* result) {
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        /* Weighted combination: 70% query, 30% context */
        result[i] = 0.7f * query[i] + 0.3f * context[i];
    }
}

/* Compute memory importance score based on multiple factors */
static float compute_memory_importance(const MemorySlot* slot, int current_iteration) {
    if (!slot || !slot->active) {
        return 0.0f;
    }
    
    /* Base importance from the slot itself */
    float base_importance = slot->importance;
    
    /* Recency factor - more recent memories are more important */
    float age = current_iteration - slot->last_accessed;
    float recency_factor = 1.0f / (1.0f + age / 10.0f);
    
    /* Access frequency factor */
    float frequency_factor = 1.0f + logf(1.0f + slot->access_count) / 10.0f;
    
    /* Associated nodes factor - memories linked to important nodes matter more */
    float node_importance_sum = 0.0f;
    for (int i = 0; i < slot->associated_node_count; i++) {
        /* This would need the model to compute properly, simplified for now */
        node_importance_sum += 0.5f;  /* Placeholder */
    }
    float node_factor = 1.0f + node_importance_sum / 10.0f;
    
    return base_importance * recency_factor * frequency_factor * node_factor;
}

/* Find least important memory slot for eviction */
static int find_lru_slot(GalileoModel* model) {
    if (!model || model->num_memory_slots == 0) {
        return -1;
    }
    
    int lru_idx = -1;
    float lowest_importance = FLT_MAX;
    
    for (int i = 0; i < model->num_memory_slots; i++) {
        MemorySlot* slot = &model->memory_slots[i];
        if (!slot->active) continue;
        
        float importance = compute_memory_importance(slot, model->current_iteration);
        
        if (importance < lowest_importance) {
            lowest_importance = importance;
            lru_idx = i;
        }
    }
    
    return lru_idx;
}

/* =============================================================================
 * ENHANCED MEMORY WRITE OPERATIONS
 * =============================================================================
 */

/* Enhanced memory write with contextual addressing and metadata */
void galileo_enhanced_memory_write(GalileoModel* model, const float* key, const float* value, 
                                   const char* description, int* associated_nodes, int node_count) {
    if (!model || !key || !value) return;
    
    /* Ensure memory module is initialized */
    if (!memory_module_initialized) {
        memory_module_init();
    }
    
    /* Find empty slot or evict least important */
    int slot_idx = -1;
    
    /* First, try to find an empty slot */
    for (int i = 0; i < MAX_MEMORY_SLOTS; i++) {
        if (!model->memory_slots[i].active) {
            slot_idx = i;
            break;
        }
    }
    
    /* If no empty slot, evict the least important */
    if (slot_idx == -1) {
        slot_idx = find_lru_slot(model);
        if (slot_idx >= 0) {
            printf("💸 Evicting memory slot %d: %s\n", 
                   slot_idx, model->memory_slots[slot_idx].description);
        } else {
            printf("⚠️  No memory slots available for eviction\n");
            return;
        }
    }
    
    MemorySlot* slot = &model->memory_slots[slot_idx];
    
    /* Copy key and value */
    memcpy(slot->key, key, EMBEDDING_DIM * sizeof(float));
    memcpy(slot->value, value, EMBEDDING_DIM * sizeof(float));
    
    /* Set description */
    if (description) {
        strncpy(slot->description, description, sizeof(slot->description) - 1);
        slot->description[sizeof(slot->description) - 1] = '\0';
    } else {
        snprintf(slot->description, sizeof(slot->description), "Memory slot %d", slot_idx);
    }
    
    /* Set metadata */
    slot->active = 1;
    slot->importance = 0.5f;  /* Default importance */
    slot->last_accessed = model->current_iteration;
    slot->access_count = 1;
    slot->created_iteration = model->current_iteration;
    
    /* Copy associated nodes */
    int nodes_to_copy = (node_count > MAX_ASSOCIATED_NODES) ? MAX_ASSOCIATED_NODES : node_count;
    if (associated_nodes && nodes_to_copy > 0) {
        memcpy(slot->associated_nodes, associated_nodes, nodes_to_copy * sizeof(int));
        slot->associated_node_count = nodes_to_copy;
        
        /* Boost importance based on node importance (simplified) */
        slot->importance += 0.1f * nodes_to_copy;
    } else {
        slot->associated_node_count = 0;
    }
    
    /* Update model memory count */
    if (slot_idx >= model->num_memory_slots) {
        model->num_memory_slots = slot_idx + 1;
    }
    
    printf("💾 Stored memory: %s (slot %d, importance %.2f)\n", 
           slot->description, slot_idx, slot->importance);
}

/* =============================================================================
 * CONTEXTUAL MEMORY RETRIEVAL
 * =============================================================================
 */

/* Contextual memory read with intelligent retrieval */
void galileo_contextual_memory_read(GalileoModel* model, const float* query, 
                                   const float* context, float* result) {
    if (!model || !query || !result) return;
    
    /* Ensure memory module is initialized */
    if (!memory_module_initialized) {
        memory_module_init();
    }
    
    /* Initialize result to zero */
    memset(result, 0, EMBEDDING_DIM * sizeof(float));
    
    if (model->num_memory_slots == 0) {
        printf("💭 No memories to retrieve from\n");
        return;
    }
    
    /* Generate contextual query key */
    float contextual_query[EMBEDDING_DIM];
    if (context) {
        generate_contextual_key(query, context, contextual_query);
    } else {
        memcpy(contextual_query, query, EMBEDDING_DIM * sizeof(float));
    }
    
    /* Find best matching memory slots */
    typedef struct {
        int slot_idx;
        float similarity;
        float importance;
        float combined_score;
    } MemoryMatch;
    
    MemoryMatch matches[MAX_MEMORY_SLOTS];
    int match_count = 0;
    
    for (int i = 0; i < model->num_memory_slots; i++) {
        MemorySlot* slot = &model->memory_slots[i];
        if (!slot->active) continue;
        
        /* Compute similarity to the contextual query */
        float similarity = cosine_similarity(contextual_query, slot->key, EMBEDDING_DIM);
        float importance = compute_memory_importance(slot, model->current_iteration);
        
        /* Combined score: similarity weighted by importance and recency */
        float combined_score = similarity * (1.0f + importance);
        
        /* Only consider reasonably similar memories */
        if (similarity > 0.1f) {
            matches[match_count].slot_idx = i;
            matches[match_count].similarity = similarity;
            matches[match_count].importance = importance;
            matches[match_count].combined_score = combined_score;
            match_count++;
        }
    }
    
    if (match_count == 0) {
        printf("💭 No relevant memories found for query\n");
        return;
    }
    
    /* Sort matches by combined score (simple bubble sort for small arrays) */
    for (int i = 0; i < match_count - 1; i++) {
        for (int j = 0; j < match_count - i - 1; j++) {
            if (matches[j].combined_score < matches[j + 1].combined_score) {
                MemoryMatch temp = matches[j];
                matches[j] = matches[j + 1];
                matches[j + 1] = temp;
            }
        }
    }
    
    /* Retrieve and blend top memories */
    int memories_to_blend = (match_count > 3) ? 3 : match_count;  /* Top 3 matches */
    float total_weight = 0.0f;
    
    printf("🔍 Retrieving %d relevant memories:\n", memories_to_blend);
    
    for (int i = 0; i < memories_to_blend; i++) {
        MemoryMatch* match = &matches[i];
        MemorySlot* slot = &model->memory_slots[match->slot_idx];
        
        float weight = match->combined_score;
        total_weight += weight;
        
        /* Blend memory value into result */
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            result[d] += weight * slot->value[d];
        }
        
        /* Update access metadata */
        slot->last_accessed = model->current_iteration;
        slot->access_count++;
        
        printf("  📁 %s (sim: %.3f, imp: %.3f, score: %.3f)\n",
               slot->description, match->similarity, match->importance, match->combined_score);
    }
    
    /* Normalize result by total weight */
    if (total_weight > 0.0f) {
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            result[d] /= total_weight;
        }
    }
    
    printf("✅ Memory retrieval complete (blended %d memories)\n", memories_to_blend);
}

/* =============================================================================
 * ADAPTIVE COMPRESSION AND SUMMARIZATION
 * =============================================================================
 */

/* Create summary node from multiple source nodes */
int galileo_create_summary_node(GalileoModel* model, int* source_nodes, int count) {
    if (!model || !source_nodes || count <= 0 || count > 10) {
        return -1;
    }
    
    if (model->num_nodes >= MAX_TOKENS) {
        printf("⚠️  Cannot create summary node: maximum nodes reached\n");
        return -1;
    }
    
    /* Create new summary node */
    int summary_idx = model->num_nodes;
    GraphNode* summary_node = &model->nodes[summary_idx];
    
    /* Initialize summary node */
    memset(summary_node, 0, sizeof(GraphNode));
    summary_node->node_id = summary_idx;
    summary_node->active = 1;
    summary_node->last_accessed_iteration = model->current_iteration;
    
    /* Generate summary embeddings by averaging source nodes */
    float total_importance = 0.0f;
    int valid_sources = 0;
    
    for (int i = 0; i < count; i++) {
        int src_idx = source_nodes[i];
        if (src_idx >= 0 && src_idx < model->num_nodes && model->nodes[src_idx].active) {
            GraphNode* src_node = &model->nodes[src_idx];
            
            /* Weighted averaging based on importance */
            float weight = src_node->importance_score;
            total_importance += weight;
            
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                summary_node->identity_embedding[d] += weight * src_node->identity_embedding[d];
                summary_node->context_embedding[d] += weight * src_node->context_embedding[d];
                summary_node->temporal_embedding[d] += weight * src_node->temporal_embedding[d];
            }
            
            valid_sources++;
        }
    }
    
    /* Normalize embeddings */
    if (total_importance > 0.0f) {
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            summary_node->identity_embedding[d] /= total_importance;
            summary_node->context_embedding[d] /= total_importance;
            summary_node->temporal_embedding[d] /= total_importance;
        }
    }
    
    /* Set summary importance as average of source importances */
    summary_node->importance_score = total_importance / valid_sources;
    
    /* Generate summary description */
    snprintf(summary_node->token_text, MAX_TOKEN_LEN, "[SUMMARY_%d_%d]", summary_idx, valid_sources);
    
    model->num_nodes++;
    
    printf("📦 Created summary node %d from %d sources (importance: %.3f)\n", 
           summary_idx, valid_sources, summary_node->importance_score);
    
    return summary_idx;
}

/* Adaptive compression based on importance and recency */
void galileo_adaptive_compression(GalileoModel* model) {
    if (!model) return;
    
    /* Ensure memory module is initialized */
    if (!memory_module_initialized) {
        memory_module_init();
    }
    
    printf("\n🗜️  === Adaptive Compression Analysis ===\n");
    
    /* Analyze nodes for compression candidates */
    typedef struct {
        int node_idx;
        float compression_score;
        int last_access_age;
        float importance;
    } CompressionCandidate;
    
    CompressionCandidate candidates[MAX_TOKENS];
    int candidate_count = 0;
    
    for (int i = 0; i < model->num_nodes; i++) {
        if (!model->nodes[i].active) continue;
        
        GraphNode* node = &model->nodes[i];
        
        /* Skip global node and recently created nodes */
        if (i == model->global_node_idx || 
            (model->current_iteration - node->last_accessed_iteration) < 3) {
            continue;
        }
        
        /* Calculate compression score based on:
         * - Low importance
         * - Age since last access
         * - Low connectivity (fewer edges)
         */
        int edge_count = 0;
        for (int e = 0; e < model->num_edges; e++) {
            if ((model->edges[e].src == i || model->edges[e].dst == i) && model->edges[e].active) {
                edge_count++;
            }
        }
        
        float age_factor = (model->current_iteration - node->last_accessed_iteration) / 5.0f;
        float connectivity_factor = 1.0f / (1.0f + edge_count);
        float compression_score = age_factor * connectivity_factor / node->importance_score;
        
        /* Only consider nodes with low compression scores */
        if (compression_score > model->compression_threshold && node->importance_score < 0.2f) {
            candidates[candidate_count].node_idx = i;
            candidates[candidate_count].compression_score = compression_score;
            candidates[candidate_count].last_access_age = model->current_iteration - node->last_accessed_iteration;
            candidates[candidate_count].importance = node->importance_score;
            candidate_count++;
        }
    }
    
    printf("🔍 Found %d compression candidates\n", candidate_count);
    
    if (candidate_count == 0) {
        printf("✅ No compression needed\n");
        return;
    }
    
    /* Group candidates for summarization (simple clustering by proximity) */
    int compressed_count = 0;
    int max_compressions = 3;  /* Limit compressions per iteration */
    
    for (int i = 0; i < candidate_count && compressed_count < max_compressions; i += 3) {
        /* Group 2-4 candidates together */
        int group_size = (candidate_count - i > 4) ? 4 : (candidate_count - i);
        if (group_size < 2) break;
        
        int group_nodes[4];
        for (int j = 0; j < group_size; j++) {
            group_nodes[j] = candidates[i + j].node_idx;
        }
        
        /* Create summary node */
        int summary_idx = galileo_create_summary_node(model, group_nodes, group_size);
        
        if (summary_idx >= 0) {
            /* Mark original nodes as compressed (deactivate) */
            for (int j = 0; j < group_size; j++) {
                model->nodes[group_nodes[j]].active = 0;
                printf("  🗜️  Compressed node %d (importance: %.3f, age: %d)\n",
                       group_nodes[j], candidates[i + j].importance, candidates[i + j].last_access_age);
            }
            
            compressed_count++;
            model->total_compressions++;
        }
    }
    
    if (compressed_count > 0) {
        printf("✅ Compressed %d node groups into summaries\n", compressed_count);
    } else {
        printf("💭 No compression performed\n");
    }
}
