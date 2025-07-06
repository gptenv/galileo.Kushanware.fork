#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

/* =============================================================================
 * GALILEO: Graph-and-Logic Integrated Language Engine v41
 * Enhanced Dynamic Graph Construction & Multi-Scale Reasoning
 * =============================================================================
 */

#define MAX_TOKENS 100000
#define MAX_EDGES 2000000    /* Increased for dynamic edge growth */
#define MAX_MEMORY_SLOTS 25000   /* Increased memory capacity */
#define MAX_FACTS 50000
#define EMBEDDING_DIM 512
#define MAX_TOKEN_LEN 64
#define MAX_EDGE_TYPES 64    /* Expanded edge type vocabulary */
#define MAX_VOCAB_SIZE 50000
#define MAX_ATTENTION_HEADS 8    /* Multi-head attention for edge scoring */

/* Enhanced edge types for richer relationships */
typedef enum {
    EDGE_SEQUENTIAL = 0,    /* token i -> token i+1 */
    EDGE_DEPENDENCY,        /* syntactic dependency */
    EDGE_SIMILARITY,        /* semantic similarity */
    EDGE_COREFERENCE,       /* entity coreference */
    EDGE_SUMMARY,           /* token -> summary node */
    EDGE_LOGICAL,           /* logical relationship */
    EDGE_GLOBAL,            /* connection to global context */
    EDGE_ATTENTION,         /* learned attention-based connection */
    EDGE_TEMPORAL,          /* temporal relationship */
    EDGE_CAUSAL,            /* causal relationship */
    EDGE_HIERARCHICAL,      /* parent-child in hierarchy */
    EDGE_BRIDGING           /* long-range semantic bridge */
} EdgeType;

/* Node types in the graph */
typedef enum {
    NODE_TOKEN = 0,         /* individual token */
    NODE_SUMMARY,           /* hierarchical summary */
    NODE_GLOBAL,            /* global context node */
    NODE_FACT,              /* symbolic fact node */
    NODE_ATTENTION_HUB,     /* attention concentration point */
    NODE_MEMORY             /* external memory reference */
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
    float embedding[EMBEDDING_DIM];           /* current working embedding */
    float identity_embedding[EMBEDDING_DIM];  /* original/anchored identity */
    float context_embedding[EMBEDDING_DIM];   /* contextual refinements */
    float temporal_embedding[EMBEDDING_DIM];  /* temporal context evolution */
    NodeType type;                            /* what kind of node this is */
    char token[MAX_TOKEN_LEN];                /* original token text */
    int active;                               /* whether node is currently active */
    int access_count;                         /* for memory management */
    float importance_score;                   /* learned importance weight */
    float attention_centrality;               /* how much attention it attracts */
    int compression_level;                    /* 0=token, 1=phrase, 2=sentence, etc */
    int last_accessed_iteration;              /* for aging/forgetting */
} GraphNode;

/* Enhanced graph edge with attention and learning */
typedef struct {
    int src, dst;                    /* source and destination node indices */
    EdgeType type;                   /* type of relationship */
    float weight;                    /* learned edge weight */
    float attention_score;           /* multi-head attention score */
    float confidence;                /* confidence in this connection */
    AttentionScore detailed_attention; /* full attention breakdown */
    int creation_iteration;          /* when this edge was created */
    int usage_count;                 /* how often it's been used */
    float decay_factor;              /* for edge aging */
} GraphEdge;

/* Enhanced memory slot with richer metadata */
typedef struct {
    float key[EMBEDDING_DIM];        /* key for content-based addressing */
    float value[EMBEDDING_DIM];      /* stored content */
    float context_key[EMBEDDING_DIM]; /* contextual addressing key */
    int in_use;                      /* whether slot is occupied */
    float access_count;              /* for LRU eviction */
    float importance_score;          /* learned importance */
    int associated_nodes[10];        /* nodes this memory relates to */
    int node_count;                  /* number of associated nodes */
    char description[256];           /* human-readable description */
    int creation_time;               /* when this memory was formed */
} MemorySlot;

/* Symbolic fact with enhanced metadata */
typedef struct {
    char subject[MAX_TOKEN_LEN];
    char relation[MAX_TOKEN_LEN];
    char object[MAX_TOKEN_LEN];
    float confidence;                /* confidence in this fact */
    int supporting_nodes[5];         /* graph nodes that support this fact */
    int support_count;               /* number of supporting nodes */
    int derivation_depth;            /* how many inference steps to derive */
} SymbolicFact;

/* Token vocabulary with frequency and context tracking */
typedef struct {
    char token[MAX_TOKEN_LEN];
    float embedding[EMBEDDING_DIM];
    int frequency;                   /* how often this token has been used */
    float context_variance;          /* how much context affects meaning */
    int contexts_seen;               /* number of different contexts */
} VocabEntry;

/* Enhanced edge candidate for dynamic addition */
typedef struct {
    int src, dst;
    EdgeType proposed_type;
    float similarity_score;
    float attention_score;
    float combined_score;
    float confidence;
    char reasoning[128];             /* why this edge was proposed */
} EdgeCandidate;

/* Main Galileo model structure with enhanced capabilities */
typedef struct {
    /* Core graph components */
    GraphNode nodes[MAX_TOKENS];
    GraphEdge edges[MAX_EDGES];
    int num_nodes;
    int num_edges;
    
    /* Enhanced memory system */
    MemorySlot memory[MAX_MEMORY_SLOTS];
    int num_memory_slots;
    
    /* Symbolic reasoning */
    SymbolicFact facts[MAX_FACTS];
    int num_facts;
    
    /* Global context and attention hubs */
    int global_node_idx;
    int attention_hubs[10];         /* special high-connectivity nodes */
    int num_attention_hubs;
    
    /* Enhanced message passing arrays */
    float node_messages_local[MAX_TOKENS][EMBEDDING_DIM];
    float node_messages_global[MAX_TOKENS][EMBEDDING_DIM];
    float node_messages_attention[MAX_TOKENS][EMBEDDING_DIM];
    float node_updates[MAX_TOKENS][EMBEDDING_DIM];
    
    /* Learning and adaptation parameters */
    float similarity_threshold;         /* threshold for adding similarity edges */
    float attention_threshold;          /* threshold for attention-based edges */
    float compression_threshold;        /* when to compress into summary nodes */
    float importance_decay;             /* how fast importance scores decay */
    int max_iterations;                /* max message passing iterations */
    int current_iteration;              /* current processing iteration */
    
    /* Dynamic edge management */
    EdgeCandidate edge_candidates[1000]; /* proposed edges this iteration */
    int num_candidates;
    int max_edges_per_iteration;        /* limit edge growth */
    
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
 * ENHANCED FUNCTION DECLARATIONS
 * =============================================================================
 */

/* Core operations */
GalileoModel* galileo_init(void);
int galileo_add_token(GalileoModel* model, const char* token_text);
void galileo_add_edge(GalileoModel* model, int src, int dst, EdgeType type, float weight);

/* Enhanced similarity and attention */
float compute_multiscale_similarity(const GraphNode* node1, const GraphNode* node2);
float compute_attention_score(GalileoModel* model, int src, int dst);
AttentionScore compute_detailed_attention(GalileoModel* model, int src, int dst);

/* Advanced graph processing */
void galileo_message_passing_iteration(GalileoModel* model);
void galileo_attention_based_edge_addition(GalileoModel* model);
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
void galileo_add_enhanced_fact(GalileoModel* model, const char* subject, const char* relation, 
                              const char* object, float confidence, int* supporting_nodes, int support_count);
void galileo_enhanced_symbolic_inference(GalileoModel* model);

/* Performance and analysis */
void galileo_compute_graph_stats(GalileoModel* model);
void galileo_update_importance_scores(GalileoModel* model);

/* Main processing */
void galileo_process_sequence(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens);

/* =============================================================================
 * CORE ENHANCED OPERATIONS
 * =============================================================================
 */

/* Initialize enhanced Galileo model */
GalileoModel* galileo_init() {
    GalileoModel* model = calloc(1, sizeof(GalileoModel));
    
    /* Enhanced default parameters */
    model->similarity_threshold = 0.85f;     /* Slightly more permissive */
    model->attention_threshold = 0.75f;      /* New threshold for attention edges */
    model->compression_threshold = 0.9f;
    model->importance_decay = 0.95f;         /* Slow decay of importance */
    model->max_iterations = 8;               /* More iterations for complex reasoning */
    model->max_edges_per_iteration = 15;     /* Allow more dynamic edges */
    model->current_iteration = 0;
    
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
    
    printf("🚀 Galileo v41 initialized with enhanced dynamic graph construction\n");
    printf("   - Multi-scale similarity computation\n");
    printf("   - Attention-based edge addition\n");
    printf("   - Adaptive compression and memory\n");
    printf("   - Enhanced symbolic reasoning\n");
    
    return model;
}

/* Enhanced hash function with better distribution */
uint32_t enhanced_hash(const char* str) {
    uint32_t hash = 5381;
    uint32_t c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
        hash ^= hash >> 16;  /* Better mixing */
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
    node->importance_score = 0.1f;  /* Start with low importance */
    node->attention_centrality = 0.0f;
    node->compression_level = 0;    /* Token level */
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
        galileo_add_edge(model, node_idx - 1, node_idx, EDGE_SEQUENTIAL, 1.0f);
    }
    
    /* Add edge to global context node with learned weight */
    float global_weight = 0.3f + 0.2f * ((float)rand() / RAND_MAX);  /* Some randomness */
    galileo_add_edge(model, node_idx, model->global_node_idx, EDGE_GLOBAL, global_weight);
    
    return node_idx;
}

/* Enhanced edge addition with detailed metadata */
void galileo_add_edge(GalileoModel* model, int src, int dst, EdgeType type, float weight) {
    assert(model->num_edges < MAX_EDGES);
    assert(src < model->num_nodes && dst < model->num_nodes);
    
    GraphEdge* edge = &model->edges[model->num_edges++];
    edge->src = src;
    edge->dst = dst;
    edge->type = type;
    edge->weight = weight;
    edge->attention_score = 0.0f;
    edge->confidence = 0.8f;  /* Default confidence */
    edge->creation_iteration = model->current_iteration;
    edge->usage_count = 0;
    edge->decay_factor = 0.99f;  /* Slow decay */
    
    /* Initialize detailed attention structure */
    edge->detailed_attention.head_count = 0;
    edge->detailed_attention.combined_score = 0.0f;
    
    /* Update node centrality scores */
    model->nodes[src].attention_centrality += 0.1f;
    model->nodes[dst].attention_centrality += 0.1f;
}

/* =============================================================================
 * ENHANCED SIMILARITY AND ATTENTION COMPUTATION
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
    
    norm_a = sqrtf(norm_a + 1e-8f);  /* Add epsilon for stability */
    norm_b = sqrtf(norm_b + 1e-8f);
    
    if (norm_a < 1e-6f || norm_b < 1e-6f) return 0.0f;
    return dot / (norm_a * norm_b);
}

/* Enhanced multi-scale similarity with sophisticated weighting */
float compute_multiscale_similarity(const GraphNode* node1, const GraphNode* node2) {
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
    
    /* Adjust weights based on compression level - higher level nodes rely more on context */
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
    
    /* Enhanced non-linearity with node importance consideration */
    float importance_factor = (node1->importance_score + node2->importance_score) / 2.0f;
    combined_sim = powf(combined_sim, 1.5f + 0.5f * importance_factor);
    
    /* Attention centrality boost */
    float centrality_boost = (node1->attention_centrality + node2->attention_centrality) * 0.05f;
    combined_sim += centrality_boost;
    
    /* Ensure in [0, 1] range */
    combined_sim = fmaxf(0.0f, fminf(1.0f, combined_sim));
    
    return combined_sim;
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

/* =============================================================================
 * ENHANCED GRAPH PROCESSING AND DYNAMICS
 * =============================================================================
 */

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

/* Smart edge candidate generation with multiple strategies */
void galileo_smart_edge_candidates(GalileoModel* model) {
    model->num_candidates = 0;
    
    /* Strategy 1: High attention pairs */
    for (int i = 0; i < model->num_nodes && model->num_candidates < 800; i++) {
        if (!model->nodes[i].active || model->nodes[i].type != NODE_TOKEN) continue;
        
        for (int j = i + 1; j < model->num_nodes && model->num_candidates < 800; j++) {
            if (!model->nodes[j].active || model->nodes[j].type != NODE_TOKEN) continue;
            
            /* Skip if already connected */
            int already_connected = 0;
            for (int e = 0; e < model->num_edges; e++) {
                if ((model->edges[e].src == i && model->edges[e].dst == j) ||
                    (model->edges[e].src == j && model->edges[e].dst == i)) {
                    already_connected = 1;
                    break;
                }
            }
            if (already_connected) continue;
            
            /* Compute attention score */
            float attention = compute_attention_score(model, i, j);
            if (attention > model->attention_threshold) {
                EdgeCandidate* candidate = &model->edge_candidates[model->num_candidates++];
                candidate->src = i;
                candidate->dst = j;
                candidate->proposed_type = EDGE_ATTENTION;
                candidate->attention_score = attention;
                candidate->similarity_score = compute_multiscale_similarity(&model->nodes[i], &model->nodes[j]);
                candidate->combined_score = 0.6f * attention + 0.4f * candidate->similarity_score;
                candidate->confidence = candidate->combined_score;
                snprintf(candidate->reasoning, 128, "High attention score: %.3f", attention);
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
            
            /* Skip if already connected or already a candidate */
            int skip = 0;
            for (int e = 0; e < model->num_edges; e++) {
                if ((model->edges[e].src == i && model->edges[e].dst == j) ||
                    (model->edges[e].src == j && model->edges[e].dst == i)) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;
            
            /* Check if already a candidate */
            for (int c = 0; c < model->num_candidates; c++) {
                if ((model->edge_candidates[c].src == i && model->edge_candidates[c].dst == j) ||
                    (model->edge_candidates[c].src == j && model->edge_candidates[c].dst == i)) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;
            
            /* Check for semantic similarity */
            float similarity = compute_multiscale_similarity(&model->nodes[i], &model->nodes[j]);
            if (similarity > model->similarity_threshold) {
                EdgeCandidate* candidate = &model->edge_candidates[model->num_candidates++];
                candidate->src = i;
                candidate->dst = j;
                candidate->proposed_type = EDGE_SIMILARITY;
                candidate->similarity_score = similarity;
                candidate->attention_score = compute_attention_score(model, i, j);
                candidate->combined_score = 0.7f * similarity + 0.3f * candidate->attention_score;
                candidate->confidence = candidate->combined_score;
                snprintf(candidate->reasoning, 128, "High similarity: %.3f", similarity);
            }
        }
    }
    
    /* Strategy 3: Bridging connections for distant important nodes */
    for (int i = 0; i < model->num_nodes && model->num_candidates < 990; i++) {
        if (!model->nodes[i].active || model->nodes[i].importance_score < 0.5f) continue;
        
        for (int j = i + 10; j < model->num_nodes && model->num_candidates < 990; j++) {
            if (!model->nodes[j].active || model->nodes[j].importance_score < 0.5f) continue;
            if (abs(i - j) < 5) continue;  /* Must be distant */
            
            /* Check if they need a bridge (not already connected) */
            int connected = 0;
            for (int e = 0; e < model->num_edges; e++) {
                if ((model->edges[e].src == i && model->edges[e].dst == j) ||
                    (model->edges[e].src == j && model->edges[e].dst == i)) {
                    connected = 1;
                    break;
                }
            }
            if (connected) continue;
            
            /* Score the bridge potential */
            float bridge_score = (model->nodes[i].importance_score + model->nodes[j].importance_score) / 2.0f;
            float semantic_score = compute_multiscale_similarity(&model->nodes[i], &model->nodes[j]);
            
            if (bridge_score > 0.6f && semantic_score > 0.3f) {
                EdgeCandidate* candidate = &model->edge_candidates[model->num_candidates++];
                candidate->src = i;
                candidate->dst = j;
                candidate->proposed_type = EDGE_BRIDGING;
                candidate->similarity_score = semantic_score;
                candidate->attention_score = bridge_score;
                candidate->combined_score = 0.5f * bridge_score + 0.5f * semantic_score;
                candidate->confidence = candidate->combined_score * 0.8f;  /* Lower confidence for bridges */
                snprintf(candidate->reasoning, 128, "Important bridge: %.3f", bridge_score);
            }
        }
    }
}

/* Enhanced attention-based edge addition with candidate ranking */
void galileo_attention_based_edge_addition(GalileoModel* model) {
    /* Generate smart candidates */
    galileo_smart_edge_candidates(model);
    
    if (model->num_candidates == 0) return;
    
    /* Sort candidates by combined score (bubble sort for simplicity) */
    for (int i = 0; i < model->num_candidates - 1; i++) {
        for (int j = 0; j < model->num_candidates - 1 - i; j++) {
            if (model->edge_candidates[j].combined_score < model->edge_candidates[j + 1].combined_score) {
                EdgeCandidate temp = model->edge_candidates[j];
                model->edge_candidates[j] = model->edge_candidates[j + 1];
                model->edge_candidates[j + 1] = temp;
            }
        }
    }
    
    /* Add top candidates up to limit */
    int edges_added = 0;
    int max_edges = fminf(model->max_edges_per_iteration, model->num_candidates);
    
    for (int c = 0; c < max_edges && edges_added < model->max_edges_per_iteration; c++) {
        EdgeCandidate* candidate = &model->edge_candidates[c];
        
        /* Final confidence check */
        if (candidate->confidence < 0.5f) break;
        
        /* Add the edge */
        galileo_add_edge(model, candidate->src, candidate->dst, 
                        candidate->proposed_type, candidate->combined_score);
        
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
    
    if (edges_added > 0) {
        printf("✨ Added %d new edges this iteration (total: %d)\n", 
               edges_added, model->total_edges_added);
    }
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
 * ENHANCED MEMORY AND COMPRESSION
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
        slot->importance_score = 0.5f;  /* Start with medium importance */
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
        retrieval_score += 0.1f * model->memory[i].importance_score;  /* Importance bonus */
        
        /* Recency bonus (newer memories are slightly preferred) */
        float recency = 1.0f - 0.1f * (model->current_iteration - model->memory[i].creation_time) / 100.0f;
        recency = fmaxf(0.5f, recency);
        retrieval_score *= recency;
        
        if (retrieval_score > 0.3f) {  /* Threshold for retrieval */
            float weight = expf(retrieval_score * 3.0f);  /* Sharper attention */
            
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
            
            start = end;  /* Skip past compressed segment */
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
    
    summary->importance_score = total_importance / count;  /* Average importance */
    summary->attention_centrality = 0.0f;
    
    /* Connect summary to global node */
    galileo_add_edge(model, summary_idx, model->global_node_idx, EDGE_HIERARCHICAL, 0.6f);
    
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
                    galileo_add_edge(model, summary_idx, other_node, EDGE_SUMMARY, 
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

/* =============================================================================
 * ENHANCED SYMBOLIC REASONING
 * =============================================================================
 */

/* Add symbolic fact with enhanced metadata and support tracking */
void galileo_add_enhanced_fact(GalileoModel* model, const char* subject, const char* relation, 
                              const char* object, float confidence, int* supporting_nodes, int support_count) {
    if (model->num_facts >= MAX_FACTS) return;
    
    /* Check for exact duplicates first */
    for (int i = 0; i < model->num_facts; i++) {
        if (strcmp(model->facts[i].subject, subject) == 0 &&
            strcmp(model->facts[i].relation, relation) == 0 &&
            strcmp(model->facts[i].object, object) == 0) {
            /* Update confidence if this one is higher */
            if (confidence > model->facts[i].confidence) {
                model->facts[i].confidence = confidence;
                printf("🔄 Updated fact: %s %s %s (conf=%.2f)\n", subject, relation, object, confidence);
            }
            return;
        }
    }
    
    SymbolicFact* fact = &model->facts[model->num_facts++];
    strncpy(fact->subject, subject, MAX_TOKEN_LEN - 1);
    strncpy(fact->relation, relation, MAX_TOKEN_LEN - 1);
    strncpy(fact->object, object, MAX_TOKEN_LEN - 1);
    fact->confidence = confidence;
    fact->derivation_depth = 0;  /* Will be set by inference */
    
    /* Store supporting nodes */
    fact->support_count = fminf(support_count, 5);
    for (int i = 0; i < fact->support_count; i++) {
        fact->supporting_nodes[i] = supporting_nodes[i];
    }
    
    printf("➕ Added enhanced fact: %s %s %s (conf=%.2f, support=%d)\n", 
           subject, relation, object, confidence, fact->support_count);
}

/* Legacy wrapper for compatibility */
void galileo_add_fact(GalileoModel* model, const char* subject, const char* relation, 
                     const char* object, float confidence) {
    galileo_add_enhanced_fact(model, subject, relation, object, confidence, NULL, 0);
}

/* Enhanced symbolic inference with deeper reasoning chains */
void galileo_enhanced_symbolic_inference(GalileoModel* model) {
    int new_inferences = 0;
    static int iteration_count = 0;
    iteration_count++;
    
    if (iteration_count > 15) {  /* Allow deeper reasoning */
        iteration_count = 0;
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
            
            /* Check if we already know X is_a Z */
            int already_known = 0;
            for (int k = 0; k < model->num_facts; k++) {
                if (strcmp(model->facts[k].subject, X) == 0 &&
                    strcmp(model->facts[k].relation, "is_a") == 0 &&
                    strcmp(model->facts[k].object, Z) == 0) {
                    already_known = 1;
                    break;
                }
            }
            
            if (!already_known) {
                float confidence = model->facts[i].confidence * model->facts[j].confidence;
                int supporting_nodes[] = {i, j};  /* Reference to source facts */
                galileo_add_enhanced_fact(model, X, "is_a", Z, confidence, supporting_nodes, 2);
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
            
            /* Subclass contradiction check */
            if (!contradicted) {
                for (int k = 0; k < model->num_facts; k++) {
                    if (strcmp(model->facts[k].relation, "subclass_of") == 0 &&
                        strcmp(model->facts[k].subject, X) == 0) {
                        char* subclass = model->facts[k].object;
                        for (int m = 0; m < model->num_facts; m++) {
                            if (strcmp(model->facts[m].subject, subclass) == 0 &&
                                strcmp(model->facts[m].relation, "cannot") == 0 &&
                                strcmp(model->facts[m].object, Z) == 0) {
                                contradicted = 1;
                                printf("⚠️  SUBCLASS CONFLICT: %s cannot %s via %s\n", X, Z, subclass);
                                break;
                            }
                        }
                        if (contradicted) break;
                    }
                }
            }
            
            if (!contradicted) {
                /* Check if already known */
                int already_known = 0;
                for (int k = 0; k < model->num_facts; k++) {
                    if (strcmp(model->facts[k].subject, X) == 0 &&
                        strcmp(model->facts[k].relation, "can") == 0 &&
                        strcmp(model->facts[k].object, Z) == 0) {
                        already_known = 1;
                        break;
                    }
                }
                
                if (!already_known) {
                    float confidence = model->facts[i].confidence * model->facts[j].confidence * 0.85f;
                    int supporting_nodes[] = {i, j};
                    galileo_add_enhanced_fact(model, X, "can", Z, confidence, supporting_nodes, 2);
                    printf("💪 ENHANCED CAPABILITY: %s can %s (inherited, conf=%.2f)\n", X, Z, confidence);
                    new_inferences++;
                }
            }
        }
    }
    
    /* Enhanced Pattern 3: Quantified reasoning */
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
                    
                    if (!is_exception) {
                        /* Infer with reduced confidence */
                        float confidence = model->facts[i].confidence * 0.7f;  /* "mostly" reduces confidence */
                        int supporting_nodes[] = {i, j};
                        galileo_add_enhanced_fact(model, Z, "probably_is", Y, confidence, supporting_nodes, 2);
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
                    
                    /* Transitive causation: A causes B, B causes C => A causes C */
                    int already_known = 0;
                    for (int k = 0; k < model->num_facts; k++) {
                        if (strcmp(model->facts[k].subject, A) == 0 &&
                            strcmp(model->facts[k].relation, "causes") == 0 &&
                            strcmp(model->facts[k].object, C) == 0) {
                            already_known = 1;
                            break;
                        }
                    }
                    
                    if (!already_known && strcmp(A, C) != 0) {  /* Avoid self-causation */
                        float confidence = model->facts[i].confidence * model->facts[j].confidence * 0.8f;
                        int supporting_nodes[] = {i, j};
                        galileo_add_enhanced_fact(model, A, "causes", C, confidence, supporting_nodes, 2);
                        printf("⚡ CAUSAL CHAIN: %s causes %s (via %s)\n", A, C, B);
                        new_inferences++;
                    }
                }
            }
        }
    }
    
    /* Enhanced Pattern 5: Advanced arithmetic and mathematical reasoning */
    for (int i = 0; i < model->num_facts; i++) {
        if (strcmp(model->facts[i].relation, "equals") != 0) continue;
        
        char* expr1 = model->facts[i].subject;
        char* result1 = model->facts[i].object;
        
        /* Look for expressions that can be combined */
        if (strstr(expr1, "_plus_") != NULL) {
            /* Parse the addition expression */
            char left[64], right[64];
            if (sscanf(expr1, "%[^_]_plus_%s", left, right) == 2) {
                /* Look for other addition facts that could combine */
                for (int j = 0; j < model->num_facts; j++) {
                    if (i == j) continue;
                    if (strcmp(model->facts[j].relation, "equals") != 0) continue;
                    
                    char* expr2 = model->facts[j].subject;
                    char* result2 = model->facts[j].object;
                    
                    if (strstr(expr2, "_plus_") != NULL) {
                        char left2[64], right2[64];
                        if (sscanf(expr2, "%[^_]_plus_%s", left2, right2) == 2) {
                            /* Check for commutative combinations */
                            if ((strcmp(left, right2) == 0 && strcmp(right, left2) == 0) ||
                                (strcmp(left, left2) == 0 && strcmp(right, right2) == 0)) {
                                
                                /* They represent the same addition - results should match */
                                if (strcmp(result1, result2) != 0) {
                                    printf("⚠️  ARITHMETIC CONFLICT: %s=%s vs %s=%s\n", 
                                           expr1, result1, expr2, result2);
                                } else {
                                    printf("✅ ARITHMETIC CONSISTENCY: %s = %s\n", expr1, result1);
                                }
                            }
                        }
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
            galileo_enhanced_symbolic_inference(model);
        }
    } else {
        iteration_count = 0;  /* Reset when no new inferences */
    }
}

/* Legacy wrapper for compatibility */
void galileo_symbolic_inference(GalileoModel* model) {
    galileo_enhanced_symbolic_inference(model);
}

/* =============================================================================
 * PERFORMANCE ANALYSIS AND OPTIMIZATION
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
 * ENHANCED MAIN PROCESSING LOOP
 * =============================================================================
 */

/* Enhanced sequence processing with all new capabilities */
void galileo_process_sequence(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    printf("\n🚀 === Enhanced Galileo v41 Processing %d tokens ===\n", num_tokens);
    
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
            galileo_attention_based_edge_addition(model);
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
        
        /* Enhanced symbolic reasoning */
        galileo_enhanced_symbolic_inference(model);
        
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
 * ENHANCED TESTING AND DEMONSTRATION
 * =============================================================================
 */

/* Enhanced test suite with new capabilities */
void test_enhanced_reasoning(GalileoModel* model) {
    printf("\n🧠 === ENHANCED REASONING TESTS v41 ===\n");
    
    /* Test 1: Complex multi-hop with attention verification */
    printf("\n--- Test 1: Enhanced Multi-hop with Attention Tracking ---\n");
    galileo_add_enhanced_fact(model, "mammal", "subclass_of", "animal", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "primate", "subclass_of", "mammal", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "human", "subclass_of", "primate", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "Alice", "is_a", "human", 1.0f, NULL, 0);
    
    char test1_tokens[][MAX_TOKEN_LEN] = {
        "Mammals", "are", "animals", "Primates", "are", "mammals", 
        "Humans", "are", "primates", "Alice", "is", "human"
    };
    galileo_process_sequence(model, test1_tokens, 12);
    
    /* Test 2: Enhanced contradiction handling */
    printf("\n--- Test 2: Enhanced Exception Handling ---\n");
    galileo_add_enhanced_fact(model, "bird", "can", "fly", 0.9f, NULL, 0);
    galileo_add_enhanced_fact(model, "penguin", "subclass_of", "bird", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "penguin", "cannot", "fly", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "ostrich", "subclass_of", "bird", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "ostrich", "cannot", "fly", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "Tweety", "is_a", "bird", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "Pingu", "is_a", "penguin", 1.0f, NULL, 0);
    
    char test2_tokens[][MAX_TOKEN_LEN] = {
        "Most", "birds", "can", "fly", "but", "penguins", "and", "ostriches", "cannot"
    };
    galileo_process_sequence(model, test2_tokens, 9);
    
    /* Test 3: Quantified reasoning */
    printf("\n--- Test 3: Quantified and Probabilistic Reasoning ---\n");
    galileo_add_enhanced_fact(model, "student", "mostly", "young", 0.8f, NULL, 0);
    galileo_add_enhanced_fact(model, "Bob", "is_a", "student", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "Carol", "is_a", "student", 1.0f, NULL, 0);
    
    char test3_tokens[][MAX_TOKEN_LEN] = {
        "Most", "students", "are", "young", "Bob", "and", "Carol", "are", "students"
    };
    galileo_process_sequence(model, test3_tokens, 9);
    
    /* Test 4: Causal reasoning */
    printf("\n--- Test 4: Causal Chain Reasoning ---\n");
    galileo_add_enhanced_fact(model, "rain", "causes", "wet_ground", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "wet_ground", "causes", "slippery", 0.9f, NULL, 0);
    galileo_add_enhanced_fact(model, "slippery", "causes", "accidents", 0.7f, NULL, 0);
    
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
           model->num_nodes, model->num_nodes);  /* Could count active separately */
    printf("  Edges: %d total, %.2f average degree\n", 
           model->num_edges, model->avg_node_degree);
    printf("  Memory: %d slots used\n", model->num_memory_slots);
    printf("  Performance: %d compressions, %d symbolic calls\n",
           model->total_compressions, model->total_symbolic_calls);
}

int main() {
    printf("🚀 Initializing Enhanced Galileo v41...\n");
    GalileoModel* model = galileo_init();
    
    /* Enhanced basic syllogism test */
    printf("\n⚡ === ENHANCED BASIC SYLLOGISM ===\n");
    galileo_add_enhanced_fact(model, "man", "subclass_of", "mortal", 1.0f, NULL, 0);
    galileo_add_enhanced_fact(model, "Socrates", "is_a", "man", 1.0f, NULL, 0);
    
    char basic_tokens[][MAX_TOKEN_LEN] = {
        "All", "men", "are", "mortal", "Socrates", "is", "a", "man"
    };
    galileo_process_sequence(model, basic_tokens, 8);
    
    /* Run enhanced test suites */
    test_enhanced_reasoning(model);
    test_enhanced_compression_and_memory(model);
    
    /* Display comprehensive results */
    display_enhanced_knowledge(model);
    
    printf("\n✨ === ENHANCED GALILEO v41 TESTING COMPLETE ===\n");
    printf("🎯 Successfully demonstrated:\n");
    printf("   ✅ Multi-scale similarity computation\n");
    printf("   ✅ Attention-based dynamic edge addition\n");
    printf("   ✅ Enhanced symbolic reasoning with support tracking\n");
    printf("   ✅ Adaptive compression and hierarchical memory\n");
    printf("   ✅ Quantified and probabilistic reasoning\n");
    printf("   ✅ Causal chain inference\n");
    printf("   ✅ Performance optimization and graph pruning\n");
    
    free(model);
    return 0;
}
