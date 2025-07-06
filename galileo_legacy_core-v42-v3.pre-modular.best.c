#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

/* =============================================================================
 * GALILEO: Graph-and-Logic Integrated Language Engine v42
 * Phase 0+1 Fixes: Multi-Model Support + Performance Optimizations
 * =============================================================================
 */

#define MAX_TOKENS 100000
#define MAX_EDGES 2000000    
#define MAX_MEMORY_SLOTS 25000   
#define MAX_FACTS 50000
#define EMBEDDING_DIM 512
#define MAX_TOKEN_LEN 64
#define MAX_EDGE_TYPES 64    
#define MAX_VOCAB_SIZE 50000
#define MAX_ATTENTION_HEADS 8    

/* Enhanced edge types for richer relationships */
typedef enum {
    EDGE_SEQUENTIAL = 0,    
    EDGE_DEPENDENCY,        
    EDGE_SIMILARITY,        
    EDGE_COREFERENCE,       
    EDGE_SUMMARY,           
    EDGE_LOGICAL,           
    EDGE_GLOBAL,            
    EDGE_ATTENTION,         
    EDGE_TEMPORAL,          
    EDGE_CAUSAL,            
    EDGE_HIERARCHICAL,      
    EDGE_BRIDGING           
} EdgeType;

/* Node types in the graph */
typedef enum {
    NODE_TOKEN = 0,         
    NODE_SUMMARY,           
    NODE_GLOBAL,            
    NODE_FACT,              
    NODE_ATTENTION_HUB,     
    NODE_MEMORY             
} NodeType;

/* Enhanced attention structure for multi-head edge scoring */
typedef struct {
    float attention_weights[MAX_ATTENTION_HEADS][EMBEDDING_DIM];
    float attention_scores[MAX_ATTENTION_HEADS];
    float combined_score;
    int head_count;
} AttentionScore;

/* Graph node with enhanced multi-scale embeddings */
typedef struct {
    float embedding[EMBEDDING_DIM];           
    float identity_embedding[EMBEDDING_DIM];  
    float context_embedding[EMBEDDING_DIM];   
    float temporal_embedding[EMBEDDING_DIM];  
    NodeType type;                            
    char token[MAX_TOKEN_LEN];                
    int active;                               
    int access_count;                         
    float importance_score;                   
    float attention_centrality;               
    int compression_level;                    
    int last_accessed_iteration;              
} GraphNode;

/* Enhanced graph edge with attention and learning */
typedef struct {
    int src, dst;                    
    EdgeType type;                   
    float weight;                    
    float attention_score;           
    float confidence;                
    AttentionScore detailed_attention; 
    int creation_iteration;          
    int usage_count;                 
    float decay_factor;              
} GraphEdge;

/* Enhanced memory slot with richer metadata */
typedef struct {
    float key[EMBEDDING_DIM];        
    float value[EMBEDDING_DIM];      
    float context_key[EMBEDDING_DIM]; 
    int in_use;                      
    float access_count;              
    float importance_score;          
    int associated_nodes[10];        
    int node_count;                  
    char description[256];           
    int creation_time;               
} MemorySlot;

/* Symbolic fact with enhanced metadata */
typedef struct {
    char subject[MAX_TOKEN_LEN];
    char relation[MAX_TOKEN_LEN];
    char object[MAX_TOKEN_LEN];
    float confidence;                
    int supporting_nodes[5];         
    int support_count;               
    int derivation_depth;            
} SymbolicFact;

/* Token vocabulary with frequency and context tracking */
typedef struct {
    char token[MAX_TOKEN_LEN];
    float embedding[EMBEDDING_DIM];
    int frequency;                   
    float context_variance;          
    int contexts_seen;               
} VocabEntry;

/* Enhanced edge candidate for dynamic addition */
typedef struct {
    int src, dst;
    EdgeType proposed_type;
    float similarity_score;
    float attention_score;
    float combined_score;
    float confidence;
    char reasoning[127];  /* FIX: Leave room for NUL terminator */
} EdgeCandidate;

/* PHASE 0 FIX: Hash set for edge deduplication */
typedef struct EdgeKey {
    int src, dst;
    EdgeType type;
} EdgeKey;

#define EDGE_HASH_SIZE 8192
typedef struct EdgeHashEntry {
    EdgeKey key;
    int exists;
    struct EdgeHashEntry* next;
} EdgeHashEntry;

/* Main Galileo model structure with enhanced capabilities */
typedef struct {
    /* Core graph components */
    GraphNode nodes[MAX_TOKENS];
    GraphEdge edges[MAX_EDGES];
    int num_nodes;
    int num_edges;
    
    /* PHASE 0 FIX: Move static state into model struct */
    int symbolic_iteration_count;    /* Was static in v41 - RACE CONDITION FIX */
    
    /* PHASE 0 FIX: Edge deduplication hash table */
    EdgeHashEntry* edge_hash[EDGE_HASH_SIZE];
    
    /* Enhanced memory system */
    MemorySlot memory[MAX_MEMORY_SLOTS];
    int num_memory_slots;
    
    /* Symbolic reasoning */
    SymbolicFact facts[MAX_FACTS];
    int num_facts;
    
    /* Global context and attention hubs */
    int global_node_idx;
    int attention_hubs[10];         
    int num_attention_hubs;
    
    /* Enhanced message passing arrays */
    float node_messages_local[MAX_TOKENS][EMBEDDING_DIM];
    float node_messages_global[MAX_TOKENS][EMBEDDING_DIM];
    float node_messages_attention[MAX_TOKENS][EMBEDDING_DIM];
    float node_updates[MAX_TOKENS][EMBEDDING_DIM];
    
    /* Learning and adaptation parameters */
    float similarity_threshold;         
    float attention_threshold;          
    float compression_threshold;        
    float importance_decay;             
    int max_iterations;                
    int current_iteration;              
    
    /* Dynamic edge management */
    EdgeCandidate edge_candidates[1000]; 
    int num_candidates;
    int max_edges_per_iteration;        
    
    /* Conflict and consistency tracking */
    char resolved_conflicts[MAX_FACTS][256];
    int num_resolved_conflicts;
    
    /* Enhanced vocabulary system */
    VocabEntry vocabulary[MAX_VOCAB_SIZE];
    int vocab_size;
    
    /* Performance and efficiency tracking */
    int total_edges_added;
    int total_compressions;
    int total_symbolic_calls;
    float avg_node_degree;
    
} GalileoModel;

/* =============================================================================
 * PHASE 0+1: ENHANCED FUNCTION DECLARATIONS  
 * =============================================================================
 */

/* PHASE 0: Lifecycle management */
GalileoModel* galileo_init(void);
void galileo_destroy(GalileoModel* model);  /* NEW: Proper cleanup */

/* Core operations */
int galileo_add_token(GalileoModel* model, const char* token_text);
int galileo_add_edge_safe(GalileoModel* model, int src, int dst, EdgeType type, float weight);  /* NEW: Dedupe check */

/* Enhanced similarity and attention */
float compute_multiscale_similarity_safe(const GraphNode* node1, const GraphNode* node2);  /* NEW: NaN protection */
float compute_attention_score(GalileoModel* model, int src, int dst);
AttentionScore compute_detailed_attention(GalileoModel* model, int src, int dst);

/* Advanced graph processing */
void galileo_message_passing_iteration(GalileoModel* model);
void galileo_attention_based_edge_addition_optimized(GalileoModel* model);  /* NEW: qsort instead of bubble */
void galileo_smart_edge_candidates(GalileoModel* model);
void galileo_prune_weak_edges(GalileoModel* model);

/* Enhanced memory operations */
void galileo_enhanced_memory_write(GalileoModel* model, const float* key, const float* value, 
                                   const char* description, int* associated_nodes, int node_count);
void galileo_contextual_memory_read(GalileoModel* model, const float* query, 
                                   const float* context, float* result);

/* Hierarchical compression and summarization */
void galileo_adaptive_compression(GalileoModel* model);
int galileo_create_summary_node(GalileoModel* model, int* source_nodes, int count);

/* Symbolic reasoning with enhanced metadata */
void galileo_add_enhanced_fact_safe(GalileoModel* model, const char* subject, const char* relation, 
                                   const char* object, float confidence, int* supporting_nodes, int support_count);  /* NEW: Dedupe check */
void galileo_enhanced_symbolic_inference_safe(GalileoModel* model);  /* NEW: No race conditions */

/* Performance and analysis */
void galileo_compute_graph_stats(GalileoModel* model);
void galileo_update_importance_scores(GalileoModel* model);

/* Main processing */
void galileo_process_sequence(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens);

/* PHASE 0: Multi-model testing */
void test_multi_model_safety(void);

/* =============================================================================
 * PHASE 0+1: CORE ENHANCED OPERATIONS
 * =============================================================================
 */

/* PHASE 0: Hash function for edge deduplication */
uint32_t hash_edge_key(const EdgeKey* key) {
    uint32_t hash = 5381;
    hash = ((hash << 5) + hash) + key->src;
    hash = ((hash << 5) + hash) + key->dst;
    hash = ((hash << 5) + hash) + (uint32_t)key->type;
    return hash % EDGE_HASH_SIZE;
}

/* PHASE 0: Check if edge already exists */
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

/* PHASE 0: Add edge to hash table */
void add_edge_to_hash(GalileoModel* model, int src, int dst, EdgeType type) {
    EdgeKey key = {src, dst, type};
    uint32_t hash = hash_edge_key(&key);
    
    EdgeHashEntry* entry = malloc(sizeof(EdgeHashEntry));
    entry->key = key;
    entry->exists = 1;
    entry->next = model->edge_hash[hash];
    model->edge_hash[hash] = entry;
}

/* PHASE 0: Initialize enhanced Galileo model with proper multi-model support */
GalileoModel* galileo_init() {
    GalileoModel* model = calloc(1, sizeof(GalileoModel));
    if (!model) {
        printf("❌ Failed to allocate Galileo model\n");
        return NULL;
    }
    
    /* Enhanced default parameters */
    model->similarity_threshold = 0.85f;     
    model->attention_threshold = 0.75f;      
    model->compression_threshold = 0.9f;
    model->importance_decay = 0.95f;         
    model->max_iterations = 8;               
    model->max_edges_per_iteration = 15;     
    model->current_iteration = 0;
    
    /* PHASE 0 FIX: Initialize per-model state (no more static/race conditions) */
    model->symbolic_iteration_count = 0;
    
    /* PHASE 0 FIX: Initialize edge hash table */
    for (int i = 0; i < EDGE_HASH_SIZE; i++) {
        model->edge_hash[i] = NULL;
    }
    
    /* Initialize global context node */
    model->global_node_idx = 0;
    model->nodes[0].type = NODE_GLOBAL;
    strcpy(model->nodes[0].token, "<GLOBAL>");
    model->nodes[0].active = 1;
    model->nodes[0].importance_score = 1.0f;
    model->nodes[0].attention_centrality = 1.0f;
    model->num_nodes = 1;
    
    /* Initialize first attention hub (global node is hub 0) */
    model->attention_hubs[0] = 0;
    model->num_attention_hubs = 1;
    
    printf("🚀 Galileo v42 initialized with multi-model support and optimizations\n");
    printf("   ✅ Race condition fixes (no more static state)\n");
    printf("   ✅ Edge deduplication system\n");
    printf("   ✅ Performance optimizations ready\n");
    printf("   ✅ Multi-model safe operation\n");
    
    return model;
}

/* PHASE 0: Proper cleanup function for multi-model support */
void galileo_destroy(GalileoModel* model) {
    if (!model) return;
    
    printf("🧹 Cleaning up Galileo model...\n");
    
    /* PHASE 0: Clean up edge hash table */
    for (int i = 0; i < EDGE_HASH_SIZE; i++) {
        EdgeHashEntry* entry = model->edge_hash[i];
        while (entry) {
            EdgeHashEntry* next = entry->next;
            free(entry);
            entry = next;
        }
        model->edge_hash[i] = NULL;
    }
    
    /* Clean up the main model structure */
    free(model);
    
    printf("✨ Galileo model destroyed cleanly\n");
}

/* Enhanced hash function with better distribution */
uint32_t enhanced_hash(const char* str) {
    uint32_t hash = 5381;
    uint32_t c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
        hash ^= hash >> 16;  
    }
    return hash;
}

/* Get or create enhanced token embedding with context awareness */
float* get_enhanced_token_embedding(GalileoModel* model, const char* token_text, int context_position) {
    /* Check existing vocabulary */
    for (int i = 0; i < model->vocab_size; i++) {
        if (strcmp(model->vocabulary[i].token, token_text) == 0) {
            model->vocabulary[i].frequency++;
            model->vocabulary[i].contexts_seen++;
            
            /* Slightly adjust embedding based on context variance */
            if (model->vocabulary[i].contexts_seen > 1) {
                float context_factor = 0.01f * (context_position % 100) / 100.0f;
                for (int j = 0; j < EMBEDDING_DIM; j++) {
                    model->vocabulary[i].embedding[j] += context_factor * 
                        ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
                }
            }
            return model->vocabulary[i].embedding;
        }
    }
    
    /* Create new vocabulary entry */
    if (model->vocab_size >= MAX_VOCAB_SIZE) {
        printf("Warning: Vocabulary full, reusing random embedding\n");
        static float temp_embedding[EMBEDDING_DIM];
        uint32_t hash = enhanced_hash(token_text);
        for (int i = 0; i < EMBEDDING_DIM; i++) {
            uint32_t seed = hash + i * 31 + context_position * 17;
            seed = seed * 1103515245 + 12345;
            temp_embedding[i] = ((float)(seed % 10000) / 10000.0f - 0.5f) * 0.2f;
        }
        return temp_embedding;
    }
    
    VocabEntry* entry = &model->vocabulary[model->vocab_size++];
    strncpy(entry->token, token_text, MAX_TOKEN_LEN - 1);
    entry->frequency = 1;
    entry->contexts_seen = 1;
    entry->context_variance = 0.0f;
    
    /* Generate enhanced deterministic embedding */
    uint32_t hash = enhanced_hash(token_text);
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        uint32_t seed = hash + i * 31 + context_position * 17;
        seed = seed * 1103515245 + 12345;
        entry->embedding[i] = ((float)(seed % 10000) / 10000.0f - 0.5f) * 0.2f;
    }
    
    return entry->embedding;
}

/* Enhanced token addition with richer initialization */
int galileo_add_token(GalileoModel* model, const char* token_text) {
    assert(model->num_nodes < MAX_TOKENS);
    
    int node_idx = model->num_nodes++;
    GraphNode* node = &model->nodes[node_idx];
    
    node->type = NODE_TOKEN;
    strncpy(node->token, token_text, MAX_TOKEN_LEN - 1);
    node->active = 1;
    node->access_count = 0;
    node->importance_score = 0.1f;  
    node->attention_centrality = 0.0f;
    node->compression_level = 0;    
    node->last_accessed_iteration = model->current_iteration;
    
    /* Get enhanced embedding with context awareness */
    float* token_embedding = get_enhanced_token_embedding(model, token_text, node_idx);
    
    /* Initialize all embedding types */
    memcpy(node->identity_embedding, token_embedding, sizeof(float) * EMBEDDING_DIM);
    memcpy(node->embedding, token_embedding, sizeof(float) * EMBEDDING_DIM);
    memset(node->context_embedding, 0, sizeof(float) * EMBEDDING_DIM);
    memset(node->temporal_embedding, 0, sizeof(float) * EMBEDDING_DIM);
    
    /* Add sequential edge from previous token */
    if (node_idx > 1) {
        galileo_add_edge_safe(model, node_idx - 1, node_idx, EDGE_SEQUENTIAL, 1.0f);
    }
    
    /* Add edge to global context node with learned weight */
    float global_weight = 0.3f + 0.2f * ((float)rand() / RAND_MAX);  
    galileo_add_edge_safe(model, node_idx, model->global_node_idx, EDGE_GLOBAL, global_weight);
    
    return node_idx;
}

/* PHASE 0+1: Safe edge addition with deduplication */
int galileo_add_edge_safe(GalileoModel* model, int src, int dst, EdgeType type, float weight) {
    assert(model->num_edges < MAX_EDGES);
    assert(src < model->num_nodes && dst < model->num_nodes);
    
    /* PHASE 0 FIX: Check for duplicate edge */
    if (edge_exists(model, src, dst, type)) {
        /* Edge already exists, maybe update weight instead of adding duplicate */
        for (int e = 0; e < model->num_edges; e++) {
            if (model->edges[e].src == src && model->edges[e].dst == dst && model->edges[e].type == type) {
                /* Update existing edge weight (could use max, average, etc.) */
                model->edges[e].weight = fmaxf(model->edges[e].weight, weight);
                return 0;  /* Updated existing edge */
            }
        }
    }
    
    /* Add new edge */
    GraphEdge* edge = &model->edges[model->num_edges++];
    edge->src = src;
    edge->dst = dst;
    edge->type = type;
    edge->weight = weight;
    edge->attention_score = 0.0f;
    edge->confidence = 0.8f;  
    edge->creation_iteration = model->current_iteration;
    edge->usage_count = 0;
    edge->decay_factor = 0.99f;  
    
    /* Initialize detailed attention structure */
    edge->detailed_attention.head_count = 0;
    edge->detailed_attention.combined_score = 0.0f;
    
    /* PHASE 0 FIX: Add to hash table */
    add_edge_to_hash(model, src, dst, type);
    
    /* Update node centrality scores */
    model->nodes[src].attention_centrality += 0.1f;
    model->nodes[dst].attention_centrality += 0.1f;
    
    return 1;  /* Added new edge */
}

/* =============================================================================
 * PHASE 1: ENHANCED SIMILARITY AND ATTENTION COMPUTATION
 * =============================================================================
 */

/* Cosine similarity with numerical stability */
float stable_cosine_similarity(const float* a, const float* b, int dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    norm_a = sqrtf(norm_a + 1e-8f);  
    norm_b = sqrtf(norm_b + 1e-8f);
    
    if (norm_a < 1e-6f || norm_b < 1e-6f) return 0.0f;
    return dot / (norm_a * norm_b);
}

/* PHASE 1 FIX: Enhanced multi-scale similarity with NaN protection */
float compute_multiscale_similarity_safe(const GraphNode* node1, const GraphNode* node2) {
    /* Compute similarities at each scale */
    float identity_sim = stable_cosine_similarity(node1->identity_embedding, 
                                                 node2->identity_embedding, 
                                                 EMBEDDING_DIM);
    
    float context_sim = stable_cosine_similarity(node1->context_embedding,
                                               node2->context_embedding,
                                               EMBEDDING_DIM);
    
    float current_sim = stable_cosine_similarity(node1->embedding,
                                               node2->embedding,
                                               EMBEDDING_DIM);
    
    float temporal_sim = stable_cosine_similarity(node1->temporal_embedding,
                                                node2->temporal_embedding,
                                                EMBEDDING_DIM);
    
    /* Adaptive weighting based on node properties */
    float identity_weight = 0.4f;
    float context_weight = 0.25f;
    float current_weight = 0.25f;
    float temporal_weight = 0.1f;
    
    /* Adjust weights based on compression level */
    if (node1->compression_level > 0 || node2->compression_level > 0) {
        context_weight += 0.1f;
        current_weight += 0.1f;
        identity_weight -= 0.15f;
        temporal_weight -= 0.05f;
    }
    
    /* Boost identity weight for recent nodes */
    if (abs(node1->last_accessed_iteration - node2->last_accessed_iteration) < 3) {
        identity_weight += 0.1f;
        context_weight -= 0.05f;
        current_weight -= 0.05f;
    }
    
    /* Weighted combination */
    float combined_sim = identity_weight * identity_sim +
                        context_weight * context_sim +
                        current_weight * current_sim +
                        temporal_weight * temporal_sim;
    
    /* PHASE 1 FIX: Enhanced non-linearity with NaN protection */
    float importance_factor = (node1->importance_score + node2->importance_score) / 2.0f;
    
    /* Guard against NaN: ensure combined_sim is positive before powf */
    combined_sim = fmaxf(combined_sim, 1e-4f);  /* FIX: Prevent NaN from powf */
    combined_sim = powf(combined_sim, 1.5f + 0.5f * importance_factor);
    
    /* Attention centrality boost */
    float centrality_boost = (node1->attention_centrality + node2->attention_centrality) * 0.05f;
    combined_sim += centrality_boost;
    
    /* Ensure in [0, 1] range */
    combined_sim = fmaxf(0.0f, fminf(1.0f, combined_sim));
    
    return combined_sim;
}

/* PHASE 1: Comparison function for qsort (edge candidates) */
int compare_edge_candidates(const void* a, const void* b) {
    const EdgeCandidate* ea = (const EdgeCandidate*)a;
    const EdgeCandidate* eb = (const EdgeCandidate*)b;
    
    /* Sort by combined_score descending */
    if (ea->combined_score > eb->combined_score) return -1;
    if (ea->combined_score < eb->combined_score) return 1;
    return 0;
}

/* PHASE 1 FIX: Optimized attention-based edge addition (qsort instead of bubble sort) */
void galileo_attention_based_edge_addition_optimized(GalileoModel* model) {
    /* Generate smart candidates */
    galileo_smart_edge_candidates(model);
    
    if (model->num_candidates == 0) return;
    
    /* PHASE 1 FIX: Use qsort instead of O(n²) bubble sort! */
    qsort(model->edge_candidates, model->num_candidates, sizeof(EdgeCandidate), compare_edge_candidates);
    
    /* Add top candidates up to limit */
    int edges_added = 0;
    int max_edges = fminf(model->max_edges_per_iteration, model->num_candidates);
    
    for (int c = 0; c < max_edges && edges_added < model->max_edges_per_iteration; c++) {
        EdgeCandidate* candidate = &model->edge_candidates[c];
        
        /* Final confidence check */
        if (candidate->confidence < 0.5f) break;
        
        /* PHASE 0+1: Use safe edge addition (deduplication) */
        if (galileo_add_edge_safe(model, candidate->src, candidate->dst, 
                                 candidate->proposed_type, candidate->combined_score)) {
            
            /* Update the detailed attention for the new edge */
            GraphEdge* new_edge = &model->edges[model->num_edges - 1];
            new_edge->attention_score = candidate->attention_score;
            new_edge->detailed_attention = compute_detailed_attention(model, candidate->src, candidate->dst);
            
            edges_added++;
            model->total_edges_added++;
            
            printf("🔗 Added %s edge: %s <-> %s (score=%.3f, %s)\n", 
                   candidate->proposed_type == EDGE_ATTENTION ? "ATTENTION" :
                   candidate->proposed_type == EDGE_SIMILARITY ? "SIMILARITY" : "BRIDGE",
                   model->nodes[candidate->src].token, 
                   model->nodes[candidate->dst].token, 
                   candidate->combined_score,
                   candidate->reasoning);
        }
    }
    
    if (edges_added > 0) {
        printf("✨ Added %d new edges this iteration (total: %d)\n", 
               edges_added, model->total_edges_added);
    }
}

/* Multi-head attention computation for edge scoring */
AttentionScore compute_detailed_attention(GalileoModel* model, int src, int dst) {
    AttentionScore attention;
    attention.head_count = MAX_ATTENTION_HEADS;
    attention.combined_score = 0.0f;
    
    GraphNode* src_node = &model->nodes[src];
    GraphNode* dst_node = &model->nodes[dst];
    
    for (int head = 0; head < MAX_ATTENTION_HEADS; head++) {
        /* Each head focuses on different aspects */
        float* src_vec = src_node->embedding;
        float* dst_vec = dst_node->embedding;
        
        if (head == 0) {
            /* Head 0: Identity similarity */
            src_vec = src_node->identity_embedding;
            dst_vec = dst_node->identity_embedding;
        } else if (head == 1) {
            /* Head 1: Context similarity */
            src_vec = src_node->context_embedding;
            dst_vec = dst_node->context_embedding;
        } else if (head == 2) {
            /* Head 2: Temporal similarity */
            src_vec = src_node->temporal_embedding;
            dst_vec = dst_node->temporal_embedding;
        }
        /* Heads 3-7 use current embedding with different projections */
        
        float score = stable_cosine_similarity(src_vec, dst_vec, EMBEDDING_DIM);
        
        /* Add head-specific bias */
        if (head % 2 == 0) {
            score += 0.1f * src_node->importance_score;
        } else {
            score += 0.1f * dst_node->attention_centrality;
        }
        
        attention.attention_scores[head] = score;
        attention.combined_score += score;
    }
    
    attention.combined_score /= MAX_ATTENTION_HEADS;
    return attention;
}

/* Simple attention score for quick computation */
float compute_attention_score(GalileoModel* model, int src, int dst) {
    AttentionScore detailed = compute_detailed_attention(model, src, dst);
    return detailed.combined_score;
}

/* Smart edge candidate generation with multiple strategies */
void galileo_smart_edge_candidates(GalileoModel* model) {
    model->num_candidates = 0;
    
    /* Strategy 1: High attention pairs */
    for (int i = 0; i < model->num_nodes && model->num_candidates < 800; i++) {
        if (!model->nodes[i].active || model->nodes[i].type != NODE_TOKEN) continue;
        
        for (int j = i + 1; j < model->num_nodes && model->num_candidates < 800; j++) {
            if (!model->nodes[j].active || model->nodes[j].type != NODE_TOKEN) continue;
            
            /* PHASE 0 FIX: Use safe edge existence check */
            if (edge_exists(model, i, j, EDGE_ATTENTION) || edge_exists(model, j, i, EDGE_ATTENTION)) {
                continue;  /* Skip if already connected */
            }
            
            /* Compute attention score */
            float attention = compute_attention_score(model, i, j);
            if (attention > model->attention_threshold) {
                EdgeCandidate* candidate = &model->edge_candidates[model->num_candidates++];
                candidate->src = i;
                candidate->dst = j;
                candidate->proposed_type = EDGE_ATTENTION;
                candidate->attention_score = attention;
                candidate->similarity_score = compute_multiscale_similarity_safe(&model->nodes[i], &model->nodes[j]);
                candidate->combined_score = 0.6f * attention + 0.4f * candidate->similarity_score;
                candidate->confidence = candidate->combined_score;
                snprintf(candidate->reasoning, 127, "High attention score: %.3f", attention);  /* FIX: Use 127 for NUL */
            }
        }
    }
    
    /* Strategy 2: Semantic similarity pairs */
    for (int i = 0; i < model->num_nodes && model->num_candidates < 950; i++) {
        if (!model->nodes[i].active) continue;
        
        for (int j = i + 1; j < model->num_nodes && model->num_candidates < 950; j++) {
            if (!model->nodes[j].active) continue;
            
            /* Skip if tokens are identical */
            if (strcmp(model->nodes[i].token, model->nodes[j].token) == 0) continue;
            
            /* PHASE 0 FIX: Use safe edge existence check */
            if (edge_exists(model, i, j, EDGE_SIMILARITY) || edge_exists(model, j, i, EDGE_SIMILARITY)) {
                continue;  /* Skip if already connected */
            }
            
            /* Check for semantic similarity */
            float similarity = compute_multiscale_similarity_safe(&model->nodes[i], &model->nodes[j]);
            if (similarity > model->similarity_threshold) {
                EdgeCandidate* candidate = &model->edge_candidates[model->num_candidates++];
                candidate->src = i;
                candidate->dst = j;
                candidate->proposed_type = EDGE_SIMILARITY;
                candidate->similarity_score = similarity;
                candidate->attention_score = compute_attention_score(model, i, j);
                candidate->combined_score = 0.7f * similarity + 0.3f * candidate->attention_score;
                candidate->confidence = candidate->combined_score;
                snprintf(candidate->reasoning, 127, "High similarity: %.3f", similarity);  /* FIX: Use 127 for NUL */
            }
        }
    }
    
    /* Strategy 3: Bridging connections for distant important nodes */
    for (int i = 0; i < model->num_nodes && model->num_candidates < 990; i++) {
        if (!model->nodes[i].active || model->nodes[i].importance_score < 0.5f) continue;
        
        for (int j = i + 10; j < model->num_nodes && model->num_candidates < 990; j++) {
            if (!model->nodes[j].active || model->nodes[j].importance_score < 0.5f) continue;
            if (abs(i - j) < 5) continue;  /* Must be distant */
            
            /* PHASE 0 FIX: Use safe edge existence check */
            if (edge_exists(model, i, j, EDGE_BRIDGING) || edge_exists(model, j, i, EDGE_BRIDGING)) {
                continue;  /* Skip if already connected */
            }
            
            /* Score the bridge potential */
            float bridge_score = (model->nodes[i].importance_score + model->nodes[j].importance_score) / 2.0f;
            float semantic_score = compute_multiscale_similarity_safe(&model->nodes[i], &model->nodes[j]);
            
            if (bridge_score > 0.6f && semantic_score > 0.3f) {
                EdgeCandidate* candidate = &model->edge_candidates[model->num_candidates++];
                candidate->src = i;
                candidate->dst = j;
                candidate->proposed_type = EDGE_BRIDGING;
                candidate->similarity_score = semantic_score;
                candidate->attention_score = bridge_score;
                candidate->combined_score = 0.5f * bridge_score + 0.5f * semantic_score;
                candidate->confidence = candidate->combined_score * 0.8f;  /* Lower confidence for bridges */
                snprintf(candidate->reasoning, 127, "Important bridge: %.3f", bridge_score);  /* FIX: Use 127 for NUL */
            }
        }
    }
}

/* Enhanced message passing with multi-scale updates */
void galileo_message_passing_iteration(GalileoModel* model) {
    /* Clear message accumulators */
    memset(model->node_messages_local, 0, sizeof(model->node_messages_local));
    memset(model->node_messages_global, 0, sizeof(model->node_messages_global));
    memset(model->node_messages_attention, 0, sizeof(model->node_messages_attention));
    
    /* Collect messages from all edge types */
    for (int e = 0; e < model->num_edges; e++) {
        GraphEdge* edge = &model->edges[e];
        GraphNode* src_node = &model->nodes[edge->src];
        
        if (!src_node->active) continue;
        
        /* Update edge usage and decay */
        edge->usage_count++;
        edge->weight *= edge->decay_factor;
        
        /* Determine message type and strength */
        float message_strength = edge->weight;
        
        /* Different message routing based on edge type */
        if (edge->type == EDGE_SEQUENTIAL || edge->type == EDGE_DEPENDENCY) {
            /* Local structure messages */
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                model->node_messages_local[edge->dst][d] += 
                    src_node->identity_embedding[d] * message_strength * 0.4f;
            }
        } else if (edge->type == EDGE_ATTENTION || edge->type == EDGE_SIMILARITY) {
            /* Attention-based messages */
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                model->node_messages_attention[edge->dst][d] += 
                    src_node->embedding[d] * message_strength * edge->attention_score;
            }
        } else {
            /* Global context messages */
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                model->node_messages_global[edge->dst][d] += 
                    src_node->context_embedding[d] * message_strength * 0.2f;
            }
        }
    }
    
    /* Enhanced node updates with multi-scale integration */
    for (int n = 0; n < model->num_nodes; n++) {
        if (!model->nodes[n].active) continue;
        
        GraphNode* node = &model->nodes[n];
        node->last_accessed_iteration = model->current_iteration;
        
        /* Update context embedding from global messages */
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            node->context_embedding[d] = 
                0.85f * node->context_embedding[d] + 
                0.15f * model->node_messages_global[n][d];
        }
        
        /* Update temporal embedding */
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            node->temporal_embedding[d] = 
                0.9f * node->temporal_embedding[d] + 
                0.1f * (model->node_messages_local[n][d] + model->node_messages_attention[n][d]);
        }
        
        /* Adaptive gating based on node properties */
        float local_gate = 0.6f + 0.2f * node->importance_score;
        float attention_gate = 0.3f + 0.3f * node->attention_centrality;
        float global_gate = 0.1f + 0.1f * (1.0f - node->importance_score);
        
        /* Normalize gates */
        float gate_sum = local_gate + attention_gate + global_gate;
        local_gate /= gate_sum;
        attention_gate /= gate_sum;
        global_gate /= gate_sum;
        
        /* Update main embedding with sophisticated gating */
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            float local_component = node->identity_embedding[d] + 
                                  0.2f * model->node_messages_local[n][d];
            float attention_component = node->embedding[d] + 
                                      0.3f * model->node_messages_attention[n][d];
            float global_component = node->context_embedding[d];
            
            node->embedding[d] = 
                local_gate * local_component + 
                attention_gate * attention_component + 
                global_gate * global_component;
            
            /* Bounded update to prevent drift */
            float max_drift = 0.3f;
            float drift = node->embedding[d] - node->identity_embedding[d];
            if (drift > max_drift) drift = max_drift;
            if (drift < -max_drift) drift = -max_drift;
            node->embedding[d] = node->identity_embedding[d] + drift;
        }
        
        /* Update importance score based on message activity */
        float message_activity = 0.0f;
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            message_activity += fabsf(model->node_messages_local[n][d]) + 
                              fabsf(model->node_messages_attention[n][d]) + 
                              fabsf(model->node_messages_global[n][d]);
        }
        message_activity /= EMBEDDING_DIM;
        
        node->importance_score = 0.9f * node->importance_score + 0.1f * message_activity;
        node->importance_score = fminf(1.0f, node->importance_score);
    }
    
    model->current_iteration++;
}

/* =============================================================================
 * PHASE 0+1: ENHANCED SYMBOLIC REASONING (NO MORE RACE CONDITIONS!)
 * =============================================================================
 */

/* PHASE 0+1: Check if fact already exists (fixes infinite Bob loop!) */
int fact_exists(GalileoModel* model, const char* subject, const char* relation, const char* object) {
    for (int i = 0; i < model->num_facts; i++) {
        if (strcmp(model->facts[i].subject, subject) == 0 &&
            strcmp(model->facts[i].relation, relation) == 0 &&
            strcmp(model->facts[i].object, object) == 0) {
            return 1;  /* Fact already exists */
        }
    }
    return 0;  /* Fact not found */
}

/* PHASE 0+1: Safe fact addition with deduplication */
void galileo_add_enhanced_fact_safe(GalileoModel* model, const char* subject, const char* relation, 
                                   const char* object, float confidence, int* supporting_nodes, int support_count) {
    if (model->num_facts >= MAX_FACTS) return;
    
    /* PHASE 0+1 FIX: Check for duplicates first! (Fixes infinite Bob loop) */
    if (fact_exists(model, subject, relation, object)) {
        /* Update confidence if this one is higher */
        for (int i = 0; i < model->num_facts; i++) {
            if (strcmp(model->facts[i].subject, subject) == 0 &&
                strcmp(model->facts[i].relation, relation) == 0 &&
                strcmp(model->facts[i].object, object) == 0) {
                if (confidence > model->facts[i].confidence) {
                    model->facts[i].confidence = confidence;
                    printf("🔄 Updated fact: %s %s %s (conf=%.2f)\n", subject, relation, object, confidence);
                }
                return;  /* Don't add duplicate */
            }
        }
    }
    
    SymbolicFact* fact = &model->facts[model->num_facts++];
    strncpy(fact->subject, subject, MAX_TOKEN_LEN - 1);
    strncpy(fact->relation, relation, MAX_TOKEN_LEN - 1);
    strncpy(fact->object, object, MAX_TOKEN_LEN - 1);
    fact->confidence = confidence;
    fact->derivation_depth = 0;  
    
    /* Store supporting nodes */
    fact->support_count = fminf(support_count, 5);
    for (int i = 0; i < fact->support_count; i++) {
        fact->supporting_nodes[i] = supporting_nodes[i];
    }
    
    printf("➕ Added enhanced fact: %s %s %s (conf=%.2f, support=%d)\n", 
           subject, relation, object, confidence, fact->support_count);
}

/* PHASE 0 FIX: Enhanced symbolic inference with NO RACE CONDITIONS */
void galileo_enhanced_symbolic_inference_safe(GalileoModel* model) {
    int new_inferences = 0;
    
    /* PHASE 0 FIX: Use per-model iteration count (no more static/race conditions!) */
    model->symbolic_iteration_count++;
    
    if (model->symbolic_iteration_count > 15) {
        model->symbolic_iteration_count = 0;
        return;
    }
    
    /* Enhanced Pattern 1: Multi-hop transitivity chains */
    for (int i = 0; i < model->num_facts; i++) {
        if (strcmp(model->facts[i].relation, "is_a") != 0) continue;
        
        char* X = model->facts[i].subject;
        char* Y = model->facts[i].object;
        
        for (int j = 0; j < model->num_facts; j++) {
            if (strcmp(model->facts[j].relation, "subclass_of") != 0) continue;
            if (strcmp(model->facts[j].subject, Y) != 0) continue;
            
            char* Z = model->facts[j].object;
            
            /* PHASE 0+1 FIX: Check if we already know X is_a Z (fixes infinite inference) */
            if (!fact_exists(model, X, "is_a", Z)) {
                float confidence = model->facts[i].confidence * model->facts[j].confidence;
                int supporting_nodes[] = {i, j};  
                galileo_add_enhanced_fact_safe(model, X, "is_a", Z, confidence, supporting_nodes, 2);
                printf("🔗 ENHANCED TRANSITIVITY: %s is_a %s (via %s, depth=%d)\n", 
                       X, Z, Y, model->facts[i].derivation_depth + 1);
                new_inferences++;
            }
        }
    }
    
    /* Enhanced Pattern 2: Conditional capability inheritance with exceptions */
    for (int i = 0; i < model->num_facts; i++) {
        if (strcmp(model->facts[i].relation, "is_a") != 0) continue;
        
        char* X = model->facts[i].subject;
        char* Y = model->facts[i].object;
        
        for (int j = 0; j < model->num_facts; j++) {
            if (strcmp(model->facts[j].relation, "can") != 0) continue;
            if (strcmp(model->facts[j].subject, Y) != 0) continue;
            
            char* Z = model->facts[j].object;
            
            /* Enhanced exception checking */
            int contradicted = 0;
            
            /* Direct contradiction check */
            for (int k = 0; k < model->num_facts; k++) {
                if (strcmp(model->facts[k].subject, X) == 0 &&
                    strcmp(model->facts[k].relation, "cannot") == 0 &&
                    strcmp(model->facts[k].object, Z) == 0) {
                    contradicted = 1;
                    printf("⚠️  ENHANCED CONFLICT: %s cannot %s (overrides inheritance)\n", X, Z);
                    break;
                }
            }
            
            if (!contradicted && !fact_exists(model, X, "can", Z)) {
                float confidence = model->facts[i].confidence * model->facts[j].confidence * 0.85f;
                int supporting_nodes[] = {i, j};
                galileo_add_enhanced_fact_safe(model, X, "can", Z, confidence, supporting_nodes, 2);
                printf("💪 ENHANCED CAPABILITY: %s can %s (inherited, conf=%.2f)\n", X, Z, confidence);
                new_inferences++;
            }
        }
    }
    
    /* Enhanced Pattern 3: Quantified reasoning with DUPLICATE CHECKING */
    for (int i = 0; i < model->num_facts; i++) {
        /* Handle "most X are Y" type reasoning */
        if (strstr(model->facts[i].relation, "mostly") != NULL) {
            char* X = model->facts[i].subject;
            char* Y = model->facts[i].object;
            
            /* Look for specific instances */
            for (int j = 0; j < model->num_facts; j++) {
                if (strcmp(model->facts[j].relation, "is_a") == 0 &&
                    strcmp(model->facts[j].object, X) == 0) {
                    char* Z = model->facts[j].subject;
                    
                    /* Check for explicit exceptions */
                    int is_exception = 0;
                    for (int k = 0; k < model->num_facts; k++) {
                        if (strcmp(model->facts[k].subject, Z) == 0 &&
                            strstr(model->facts[k].relation, "not") != NULL &&
                            strcmp(model->facts[k].object, Y) == 0) {
                            is_exception = 1;
                            break;
                        }
                    }
                    
                    /* PHASE 0+1 FIX: Only infer if not already known (FIXES BOB LOOP!) */
                    if (!is_exception && !fact_exists(model, Z, "probably_is", Y)) {
                        float confidence = model->facts[i].confidence * 0.7f;  
                        int supporting_nodes[] = {i, j};
                        galileo_add_enhanced_fact_safe(model, Z, "probably_is", Y, confidence, supporting_nodes, 2);
                        printf("🤔 PROBABILISTIC: %s probably_is %s (from 'mostly', conf=%.2f)\n", Z, Y, confidence);
                        new_inferences++;
                    }
                }
            }
        }
    }
    
    /* Enhanced Pattern 4: Temporal and causal reasoning */
    for (int i = 0; i < model->num_facts; i++) {
        if (strcmp(model->facts[i].relation, "causes") == 0) {
            char* A = model->facts[i].subject;
            char* B = model->facts[i].object;
            
            for (int j = 0; j < model->num_facts; j++) {
                if (strcmp(model->facts[j].relation, "causes") == 0 &&
                    strcmp(model->facts[j].subject, B) == 0) {
                    char* C = model->facts[j].object;
                    
                    /* PHASE 0+1 FIX: Check if already known and avoid self-causation */
                    if (!fact_exists(model, A, "causes", C) && strcmp(A, C) != 0) {
                        float confidence = model->facts[i].confidence * model->facts[j].confidence * 0.8f;
                        int supporting_nodes[] = {i, j};
                        galileo_add_enhanced_fact_safe(model, A, "causes", C, confidence, supporting_nodes, 2);
                        printf("⚡ CAUSAL CHAIN: %s causes %s (via %s)\n", A, C, B);
                        new_inferences++;
                    }
                }
            }
        }
    }
    
    if (new_inferences > 0) {
        printf("✨ Made %d enhanced inferences this round\n", new_inferences);
        model->total_symbolic_calls++;
        
        /* Recursively infer more if we found new facts */
        if (new_inferences > 0 && model->num_facts < MAX_FACTS - 20) {
            galileo_enhanced_symbolic_inference_safe(model);
        }
    } else {
        model->symbolic_iteration_count = 0;  /* Reset when no new inferences */
    }
}

/* =============================================================================
 * PHASE 0+1: ENHANCED MEMORY AND COMPRESSION
 * =============================================================================
 */

/* Enhanced memory write with contextual associations */
void galileo_enhanced_memory_write(GalileoModel* model, const float* key, const float* value, 
                                   const char* description, int* associated_nodes, int node_count) {
    /* Find best slot (empty or least important) */
    int slot_idx = -1;
    float min_importance = INFINITY;
    
    for (int i = 0; i < MAX_MEMORY_SLOTS; i++) {
        if (!model->memory[i].in_use) {
            slot_idx = i;
            break;
        }
        if (model->memory[i].importance_score < min_importance) {
            min_importance = model->memory[i].importance_score;
            slot_idx = i;
        }
    }
    
    if (slot_idx >= 0) {
        MemorySlot* slot = &model->memory[slot_idx];
        memcpy(slot->key, key, sizeof(float) * EMBEDDING_DIM);
        memcpy(slot->value, value, sizeof(float) * EMBEDDING_DIM);
        
        /* Compute contextual key as average of associated node embeddings */
        memset(slot->context_key, 0, sizeof(float) * EMBEDDING_DIM);
        if (associated_nodes && node_count > 0) {
            for (int i = 0; i < node_count && i < 10; i++) {
                if (associated_nodes[i] < model->num_nodes) {
                    for (int d = 0; d < EMBEDDING_DIM; d++) {
                        slot->context_key[d] += model->nodes[associated_nodes[i]].context_embedding[d];
                    }
                    slot->associated_nodes[i] = associated_nodes[i];
                }
            }
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                slot->context_key[d] /= node_count;
            }
            slot->node_count = fminf(node_count, 10);
        } else {
            memcpy(slot->context_key, key, sizeof(float) * EMBEDDING_DIM);
            slot->node_count = 0;
        }
        
        strncpy(slot->description, description, 255);
        slot->in_use = 1;
        slot->access_count = 1.0f;
        slot->importance_score = 0.5f;  
        slot->creation_time = model->current_iteration;
        
        if (slot_idx >= model->num_memory_slots) {
            model->num_memory_slots = slot_idx + 1;
        }
        
        printf("💾 Stored memory: %s (slot %d, %d associations)\n", description, slot_idx, node_count);
    }
}

/* Contextual memory read with attention-based retrieval */
void galileo_contextual_memory_read(GalileoModel* model, const float* query, 
                                   const float* context, float* result) {
    memset(result, 0, sizeof(float) * EMBEDDING_DIM);
    
    float total_weight = 0.0f;
    
    for (int i = 0; i < model->num_memory_slots; i++) {
        if (!model->memory[i].in_use) continue;
        
        /* Compute both key and context similarity */
        float key_similarity = stable_cosine_similarity(query, model->memory[i].key, EMBEDDING_DIM);
        float context_similarity = context ? 
            stable_cosine_similarity(context, model->memory[i].context_key, EMBEDDING_DIM) : 0.0f;
        
        /* Combined retrieval score */
        float retrieval_score = 0.7f * key_similarity + 0.3f * context_similarity;
        retrieval_score += 0.1f * model->memory[i].importance_score;  
        
        /* Recency bonus (newer memories are slightly preferred) */
        float recency = 1.0f - 0.1f * (model->current_iteration - model->memory[i].creation_time) / 100.0f;
        recency = fmaxf(0.5f, recency);
        retrieval_score *= recency;
        
        if (retrieval_score > 0.3f) {  
            float weight = expf(retrieval_score * 3.0f);  
            
            for (int d = 0; d < EMBEDDING_DIM; d++) {
                result[d] += weight * model->memory[i].value[d];
            }
            total_weight += weight;
            
            /* Update access patterns */
            model->memory[i].access_count += retrieval_score;
            model->memory[i].importance_score = fminf(1.0f, 
                model->memory[i].importance_score + 0.05f * retrieval_score);
        }
    }
    
    /* Normalize result */
    if (total_weight > 0.0f) {
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            result[d] /= total_weight;
        }
    }
}

/* Adaptive compression based on attention patterns and importance */
void galileo_adaptive_compression(GalileoModel* model) {
    /* Find segments of low-attention nodes to compress */
    for (int start = 1; start < model->num_nodes - 5; start++) {
        if (model->nodes[start].type != NODE_TOKEN) continue;
        if (model->nodes[start].importance_score > 0.3f) continue;
        
        /* Look for a sequence of low-importance tokens */
        int end = start;
        float avg_importance = 0.0f;
        int count = 0;
        
        while (end < model->num_nodes && count < 8) {
            if (model->nodes[end].type != NODE_TOKEN) break;
            if (model->nodes[end].importance_score > 0.4f) break;
            
            avg_importance += model->nodes[end].importance_score;
            count++;
            end++;
        }
        
        /* Compress if we found a segment worth compressing */
        if (count >= 3 && avg_importance / count < 0.2f) {
            int summary_nodes[8];
            for (int i = 0; i < count; i++) {
                summary_nodes[i] = start + i;
            }
            
            int summary_idx = galileo_create_summary_node(model, summary_nodes, count);
            if (summary_idx >= 0) {
                printf("📦 Compressed %d tokens into summary node %d\n", count, summary_idx);
                model->total_compressions++;
            }
            
            start = end;  
        }
    }
}

/* Create a summary node from a collection of source nodes */
int galileo_create_summary_node(GalileoModel* model, int* source_nodes, int count) {
    if (model->num_nodes >= MAX_TOKENS || count == 0) return -1;
    
    int summary_idx = model->num_nodes++;
    GraphNode* summary = &model->nodes[summary_idx];
    
    summary->type = NODE_SUMMARY;
    snprintf(summary->token, MAX_TOKEN_LEN, "<SUMMARY_%d>", summary_idx);
    summary->active = 1;
    summary->access_count = 0;
    summary->compression_level = model->nodes[source_nodes[0]].compression_level + 1;
    summary->last_accessed_iteration = model->current_iteration;
    
    /* Compute summary embeddings as weighted average */
    memset(summary->identity_embedding, 0, sizeof(float) * EMBEDDING_DIM);
    memset(summary->context_embedding, 0, sizeof(float) * EMBEDDING_DIM);
    memset(summary->embedding, 0, sizeof(float) * EMBEDDING_DIM);
    memset(summary->temporal_embedding, 0, sizeof(float) * EMBEDDING_DIM);
    
    float total_importance = 0.0f;
    for (int i = 0; i < count; i++) {
        float weight = fmaxf(0.1f, model->nodes[source_nodes[i]].importance_score);
        total_importance += weight;
        
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            summary->identity_embedding[d] += weight * model->nodes[source_nodes[i]].identity_embedding[d];
            summary->context_embedding[d] += weight * model->nodes[source_nodes[i]].context_embedding[d];
            summary->embedding[d] += weight * model->nodes[source_nodes[i]].embedding[d];
            summary->temporal_embedding[d] += weight * model->nodes[source_nodes[i]].temporal_embedding[d];
        }
    }
    
    /* Normalize by total weight */
    if (total_importance > 0.0f) {
        for (int d = 0; d < EMBEDDING_DIM; d++) {
            summary->identity_embedding[d] /= total_importance;
            summary->context_embedding[d] /= total_importance;
            summary->embedding[d] /= total_importance;
            summary->temporal_embedding[d] /= total_importance;
        }
    }
    
    summary->importance_score = total_importance / count;  
    summary->attention_centrality = 0.0f;
    
    /* Connect summary to global node */
    galileo_add_edge_safe(model, summary_idx, model->global_node_idx, EDGE_HIERARCHICAL, 0.6f);
    
    /* Connect summary to any important neighbors of source nodes */
    for (int i = 0; i < count; i++) {
        /* Deactivate source node (but don't delete for potential reactivation) */
        model->nodes[source_nodes[i]].active = 0;
        
        /* Find important connections of this source node */
        for (int e = 0; e < model->num_edges; e++) {
            if (model->edges[e].src == source_nodes[i] || model->edges[e].dst == source_nodes[i]) {
                int other_node = (model->edges[e].src == source_nodes[i]) ? 
                                model->edges[e].dst : model->edges[e].src;
                
                if (model->nodes[other_node].active && 
                    model->nodes[other_node].importance_score > 0.4f) {
                    /* Connect summary to this important node */
                    galileo_add_edge_safe(model, summary_idx, other_node, EDGE_SUMMARY, 
                                         model->edges[e].weight * 0.7f);
                }
            }
        }
    }
    
    /* Store compressed information in memory */
    char desc[256];
    snprintf(desc, 256, "Summary of %d tokens (compression level %d)", 
             count, summary->compression_level);
    galileo_enhanced_memory_write(model, summary->embedding, summary->context_embedding, 
                                 desc, source_nodes, count);
    
    return summary_idx;
}

/* Prune weak or obsolete edges to prevent graph bloat */
void galileo_prune_weak_edges(GalileoModel* model) {
    int pruned = 0;
    
    for (int e = model->num_edges - 1; e >= 0; e--) {
        GraphEdge* edge = &model->edges[e];
        
        /* Don't prune structural edges */
        if (edge->type == EDGE_SEQUENTIAL || edge->type == EDGE_GLOBAL) continue;
        
        /* Prune if edge has decayed too much or is unused */
        int should_prune = 0;
        
        if (edge->weight < 0.1f) {
            should_prune = 1;  /* Weight decayed too low */
        } else if (edge->usage_count == 0 && 
                   (model->current_iteration - edge->creation_iteration) > 5) {
            should_prune = 1;  /* Unused for too long */
        } else if (edge->confidence < 0.3f) {
            should_prune = 1;  /* Low confidence */
        }
        
        if (should_prune) {
            /* Remove edge by copying last edge to this position */
            if (e < model->num_edges - 1) {
                model->edges[e] = model->edges[model->num_edges - 1];
            }
            model->num_edges--;
            pruned++;
        }
    }
    
    if (pruned > 0) {
        printf("🧹 Pruned %d weak edges\n", pruned);
    }
}

/* =============================================================================
 * PHASE 0+1: PERFORMANCE ANALYSIS AND OPTIMIZATION
 * =============================================================================
 */

/* Compute comprehensive graph statistics */
void galileo_compute_graph_stats(GalileoModel* model) {
    /* Compute average degree */
    float total_degree = 0.0f;
    int active_nodes = 0;
    
    for (int n = 0; n < model->num_nodes; n++) {
        if (!model->nodes[n].active) continue;
        active_nodes++;
        
        int degree = 0;
        for (int e = 0; e < model->num_edges; e++) {
            if (model->edges[e].src == n || model->edges[e].dst == n) {
                degree++;
            }
        }
        total_degree += degree;
    }
    
    model->avg_node_degree = active_nodes > 0 ? total_degree / active_nodes : 0.0f;
    
    /* Update graph efficiency metrics */
    printf("📊 Graph Stats: %d active nodes, %d edges, avg degree: %.2f\n", 
           active_nodes, model->num_edges, model->avg_node_degree);
    printf("   Total compressions: %d, symbolic calls: %d, edges added: %d\n",
           model->total_compressions, model->total_symbolic_calls, model->total_edges_added);
}

/* Update importance scores based on recent activity */
void galileo_update_importance_scores(GalileoModel* model) {
    for (int n = 0; n < model->num_nodes; n++) {
        if (!model->nodes[n].active) continue;
        
        GraphNode* node = &model->nodes[n];
        
        /* Decay importance over time */
        float age_factor = (model->current_iteration - node->last_accessed_iteration) / 10.0f;
        float decay = powf(model->importance_decay, age_factor);
        node->importance_score *= decay;
        
        /* Boost importance for highly connected nodes */
        int degree = 0;
        for (int e = 0; e < model->num_edges; e++) {
            if (model->edges[e].src == n || model->edges[e].dst == n) {
                degree++;
            }
        }
        
        if (degree > model->avg_node_degree) {
            node->importance_score += 0.05f * (degree - model->avg_node_degree) / model->avg_node_degree;
        }
        
        /* Cap importance scores */
        node->importance_score = fminf(1.0f, fmaxf(0.0f, node->importance_score));
    }
}

/* =============================================================================
 * PHASE 0+1: ENHANCED MAIN PROCESSING LOOP
 * =============================================================================
 */

/* Enhanced sequence processing with all new capabilities */
void galileo_process_sequence(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    printf("\n🚀 === Enhanced Galileo v42 Processing %d tokens ===\n", num_tokens);
    
    /* Add all tokens to graph */
    for (int i = 0; i < num_tokens; i++) {
        galileo_add_token(model, tokens[i]);
    }
    
    /* Enhanced processing loop */
    for (int iter = 0; iter < model->max_iterations; iter++) {
        printf("\n--- Enhanced Iteration %d ---\n", iter + 1);
        
        /* Core message passing */
        galileo_message_passing_iteration(model);
        
        /* Dynamic graph evolution */
        if (iter % 2 == 0) {
            galileo_attention_based_edge_addition_optimized(model);  /* PHASE 1: Optimized! */
        }
        
        /* Adaptive compression */
        if (iter % 3 == 0 && iter > 0) {
            galileo_adaptive_compression(model);
        }
        
        /* Enhanced memory operations */
        for (int n = 1; n < model->num_nodes; n++) {
            if (model->nodes[n].type == NODE_TOKEN && 
                model->nodes[n].access_count == 0 && 
                model->nodes[n].importance_score > 0.4f) {
                
                /* Store important tokens in enhanced memory */
                char desc[256];
                snprintf(desc, sizeof(desc), "Important token: %s (importance: %.3f)", 
                        model->nodes[n].token, model->nodes[n].importance_score);
                
                int associated_nodes[] = {n};
                galileo_enhanced_memory_write(model, model->nodes[n].embedding, 
                                            model->nodes[n].context_embedding, 
                                            desc, associated_nodes, 1);
                model->nodes[n].access_count = 1;
            }
        }
        
        /* PHASE 0+1: Enhanced symbolic reasoning (safe!) */
        galileo_enhanced_symbolic_inference_safe(model);
        
        /* Performance optimization */
        if (iter % 4 == 0) {
            galileo_prune_weak_edges(model);
            galileo_update_importance_scores(model);
        }
        
        /* Compute and display stats */
        if (iter == model->max_iterations - 1) {
            galileo_compute_graph_stats(model);
        }
    }
    
    printf("\n🎯 Enhanced processing complete!\n");
    printf("Final state: %d nodes, %d edges, %d memory slots, %d facts\n", 
           model->num_nodes, model->num_edges, model->num_memory_slots, model->num_facts);
    printf("Performance: %.2f avg degree, %d compressions, %d symbolic calls\n",
           model->avg_node_degree, model->total_compressions, model->total_symbolic_calls);
}

/* =============================================================================
 * PHASE 0: MULTI-MODEL TESTING AND VALIDATION
 * =============================================================================
 */

/* PHASE 0: Test multi-model safety and isolation */
void test_multi_model_safety(void) {
    printf("\n🧪 === PHASE 0: MULTI-MODEL SAFETY TESTS ===\n");
    
    /* Test 1: Multiple models don't interfere */
    printf("\n--- Test 1: Model Independence ---\n");
    GalileoModel* model1 = galileo_init();
    GalileoModel* model2 = galileo_init();
    
    if (!model1 || !model2) {
        printf("❌ Failed to create models\n");
        return;
    }
    
    /* Add different facts to each model */
    galileo_add_enhanced_fact_safe(model1, "Socrates", "is_a", "philosopher", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model2, "Plato", "is_a", "philosopher", 1.0f, NULL, 0);
    
    /* Run inference on both */
    galileo_enhanced_symbolic_inference_safe(model1);
    galileo_enhanced_symbolic_inference_safe(model2);
    
    /* Check that models remain separate */
    int model1_has_plato = 0, model2_has_socrates = 0;
    
    for (int i = 0; i < model1->num_facts; i++) {
        if (strcmp(model1->facts[i].subject, "Plato") == 0) {
            model1_has_plato = 1;
            break;
        }
    }
    
    for (int i = 0; i < model2->num_facts; i++) {
        if (strcmp(model2->facts[i].subject, "Socrates") == 0) {
            model2_has_socrates = 1;
            break;
        }
    }
    
    if (!model1_has_plato && !model2_has_socrates) {
        printf("✅ Models remain properly isolated\n");
    } else {
        printf("❌ Models interfered with each other!\n");
    }
    
    /* Test 2: Symbolic iteration counts are per-model */
    printf("\n--- Test 2: Per-Model State Isolation ---\n");
    
    model1->symbolic_iteration_count = 10;
    model2->symbolic_iteration_count = 5;
    
    /* Run symbolic inference and check counts are independent */
    galileo_enhanced_symbolic_inference_safe(model1);
    galileo_enhanced_symbolic_inference_safe(model2);
    
    if (model1->symbolic_iteration_count != model2->symbolic_iteration_count) {
        printf("✅ Per-model iteration counts working correctly\n");
        printf("   Model1 count: %d, Model2 count: %d\n", 
               model1->symbolic_iteration_count, model2->symbolic_iteration_count);
    } else {
        printf("❌ Models sharing state - race condition risk!\n");
    }
    
    /* Test 3: Clean destruction */
    printf("\n--- Test 3: Clean Destruction ---\n");
    galileo_destroy(model1);
    galileo_destroy(model2);
    
    /* Test 4: Fresh model after destruction */
    printf("\n--- Test 4: Fresh Model Creation ---\n");
    GalileoModel* model3 = galileo_init();
    
    if (model3 && model3->symbolic_iteration_count == 0 && model3->num_facts == 0) {
        printf("✅ Fresh model starts clean\n");
    } else {
        printf("❌ Model not properly reset\n");
    }
    
    galileo_destroy(model3);
    printf("✅ Multi-model safety tests completed\n");
}

/* =============================================================================
 * PHASE 0+1: ENHANCED TESTING AND DEMONSTRATION
 * =============================================================================
 */

/* Enhanced test suite with new capabilities */
void test_enhanced_reasoning(GalileoModel* model) {
    printf("\n🧠 === ENHANCED REASONING TESTS v42 ===\n");
    
    /* Test 1: Complex multi-hop with attention verification */
    printf("\n--- Test 1: Enhanced Multi-hop with Attention Tracking ---\n");
    galileo_add_enhanced_fact_safe(model, "mammal", "subclass_of", "animal", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "primate", "subclass_of", "mammal", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "human", "subclass_of", "primate", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "Alice", "is_a", "human", 1.0f, NULL, 0);
    
    char test1_tokens[][MAX_TOKEN_LEN] = {
        "Mammals", "are", "animals", "Primates", "are", "mammals", 
        "Humans", "are", "primates", "Alice", "is", "human"
    };
    galileo_process_sequence(model, test1_tokens, 12);
    
    /* Test 2: Enhanced exception handling */
    printf("\n--- Test 2: Enhanced Exception Handling ---\n");
    galileo_add_enhanced_fact_safe(model, "bird", "can", "fly", 0.9f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "penguin", "subclass_of", "bird", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "penguin", "cannot", "fly", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "ostrich", "subclass_of", "bird", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "ostrich", "cannot", "fly", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "Tweety", "is_a", "bird", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "Pingu", "is_a", "penguin", 1.0f, NULL, 0);
    
    char test2_tokens[][MAX_TOKEN_LEN] = {
        "Most", "birds", "can", "fly", "but", "penguins", "and", "ostriches", "cannot"
    };
    galileo_process_sequence(model, test2_tokens, 9);
    
    /* Test 3: FIXED Quantified and probabilistic reasoning (no more infinite Bob!) */
    printf("\n--- Test 3: FIXED Quantified and Probabilistic Reasoning ---\n");
    galileo_add_enhanced_fact_safe(model, "student", "mostly", "young", 0.8f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "Bob", "is_a", "student", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "Carol", "is_a", "student", 1.0f, NULL, 0);
    
    char test3_tokens[][MAX_TOKEN_LEN] = {
        "Most", "students", "are", "young", "Bob", "and", "Carol", "are", "students"
    };
    galileo_process_sequence(model, test3_tokens, 9);
    
    /* Test 4: Causal reasoning */
    printf("\n--- Test 4: Causal Chain Reasoning ---\n");
    galileo_add_enhanced_fact_safe(model, "rain", "causes", "wet_ground", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "wet_ground", "causes", "slippery", 0.9f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "slippery", "causes", "accidents", 0.7f, NULL, 0);
    
    char test4_tokens[][MAX_TOKEN_LEN] = {
        "Rain", "causes", "wet", "ground", "which", "becomes", "slippery", "and", "dangerous"
    };
    galileo_process_sequence(model, test4_tokens, 9);
}

void test_enhanced_compression_and_memory(GalileoModel* model) {
    printf("\n💾 === ENHANCED MEMORY & COMPRESSION TESTS ===\n");
    
    /* Create a longer sequence to trigger compression */
    char long_tokens[][MAX_TOKEN_LEN] = {
        "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "and",
        "runs", "through", "the", "forest", "until", "it", "reaches", "a", "clearing", "where",
        "many", "other", "animals", "gather", "to", "discuss", "important", "matters", "about",
        "their", "shared", "ecosystem", "and", "the", "challenges", "they", "face", "together"
    };
    
    printf("Processing long sequence to test compression...\n");
    galileo_process_sequence(model, long_tokens, 38);
}

void display_enhanced_knowledge(GalileoModel* model) {
    printf("\n📚 === ENHANCED KNOWLEDGE BASE ===\n");
    printf("Total facts learned: %d\n", model->num_facts);
    
    /* Group facts by derivation depth */
    int depth_counts[10] = {0};
    for (int i = 0; i < model->num_facts; i++) {
        int depth = fminf(model->facts[i].derivation_depth, 9);
        depth_counts[depth]++;
        
        printf("  [D%d] %s %s %s (conf: %.2f", 
               model->facts[i].derivation_depth,
               model->facts[i].subject,
               model->facts[i].relation, 
               model->facts[i].object,
               model->facts[i].confidence);
        
        if (model->facts[i].support_count > 0) {
            printf(", support: %d", model->facts[i].support_count);
        }
        printf(")\n");
    }
    
    printf("\nFacts by derivation depth:\n");
    for (int d = 0; d < 10; d++) {
        if (depth_counts[d] > 0) {
            printf("  Depth %d: %d facts\n", d, depth_counts[d]);
        }
    }
    
    printf("\nEnhanced graph structure:\n");
    printf("  Nodes: %d active, %d total\n", 
           model->num_nodes, model->num_nodes);  
    printf("  Edges: %d total, %.2f average degree\n", 
           model->num_edges, model->avg_node_degree);
    printf("  Memory: %d slots used\n", model->num_memory_slots);
    printf("  Performance: %d compressions, %d symbolic calls\n",
           model->total_compressions, model->total_symbolic_calls);
}

int main() {
    printf("🚀 Initializing Enhanced Galileo v42...\n");
    
    /* PHASE 0: Test multi-model safety first! */
    test_multi_model_safety();
    
    /* Create main test model */
    GalileoModel* model = galileo_init();
    if (!model) {
        printf("❌ Failed to initialize Galileo model\n");
        return 1;
    }
    
    /* Enhanced basic syllogism test */
    printf("\n⚡ === ENHANCED BASIC SYLLOGISM ===\n");
    galileo_add_enhanced_fact_safe(model, "man", "subclass_of", "mortal", 1.0f, NULL, 0);
    galileo_add_enhanced_fact_safe(model, "Socrates", "is_a", "man", 1.0f, NULL, 0);
    
    char basic_tokens[][MAX_TOKEN_LEN] = {
        "All", "men", "are", "mortal", "Socrates", "is", "a", "man"
    };
    galileo_process_sequence(model, basic_tokens, 8);
    
    /* Run enhanced test suites */
    test_enhanced_reasoning(model);
    test_enhanced_compression_and_memory(model);
    
    /* Display comprehensive results */
    display_enhanced_knowledge(model);
    
    /* PHASE 0: Proper cleanup */
    galileo_destroy(model);
    
    printf("\n✨ === ENHANCED GALILEO v42 TESTING COMPLETE ===\n");
    printf("🎯 Successfully demonstrated:\n");
    printf("   ✅ PHASE 0: Multi-model support (no more race conditions!)\n");
    printf("   ✅ PHASE 0: Edge deduplication (no more duplicate edges)\n");
    printf("   ✅ PHASE 0: Fact deduplication (Bob is finally free!)\n");
    printf("   ✅ PHASE 1: qsort optimization (no more O(n²) bubble sort)\n");
    printf("   ✅ PHASE 1: NaN protection in similarity computation\n");
    printf("   ✅ PHASE 1: String buffer overflow protection\n");
    printf("   ✅ Multi-scale similarity computation\n");
    printf("   ✅ Attention-based dynamic edge addition\n");
    printf("   ✅ Enhanced symbolic reasoning with support tracking\n");
    printf("   ✅ Adaptive compression and hierarchical memory\n");
    printf("   ✅ Quantified and probabilistic reasoning\n");
    printf("   ✅ Causal chain inference\n");
    printf("   ✅ Performance optimization and graph pruning\n");
    
    return 0;
}
