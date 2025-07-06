/* =============================================================================
 * galileo/src/core/galileo_types.h - Complete Type Definitions (JANITOR FIXED)
 * 
 * CLAUDE JANITOR CLEANUP: Added ALL missing fields that other modules use!
 * This header now contains EVERY field that's actually referenced in the code.
 * 
 * Previous Claude left a mess - types header missing tons of fields! 🧹
 * =============================================================================
 */

#ifndef GALILEO_TYPES_H
#define GALILEO_TYPES_H

#include <stdint.h>
#include <stddef.h>  /* For size_t */
#include <time.h>    /* For time_t - needed for first_seen_time field */

/* =============================================================================
 * FUNDAMENTAL CONSTANTS
 * =============================================================================
 */

/* Version information */
#ifndef GALILEO_VERSION
#define GALILEO_VERSION "42"
#endif

/* Token and text limits */
#ifndef MAX_TOKEN_LEN
#define MAX_TOKEN_LEN 64
#endif

/* Graph structure limits */
#ifndef MAX_TOKENS
#define MAX_TOKENS 100000  /* Increased for scalability */
#endif

#ifndef MAX_EDGES
#define MAX_EDGES 2000000  /* Massive increase for dynamic growth */
#endif

/* Memory system limits */
#ifndef MAX_MEMORY_SLOTS
#define MAX_MEMORY_SLOTS 25000  /* Increased capacity */
#endif

/* Symbolic reasoning limits */
#ifndef MAX_FACTS
#define MAX_FACTS 50000  /* Increased for complex reasoning */
#endif

/* Vocabulary management */
#ifndef MAX_VOCAB_SIZE
#define MAX_VOCAB_SIZE 50000  /* Much larger vocabulary */
#endif

/* Neural network dimensions */
#ifndef EMBEDDING_DIM
#define EMBEDDING_DIM 512
#endif

/* Multi-head attention */
#ifndef MAX_ATTENTION_HEADS
#define MAX_ATTENTION_HEADS 8
#endif

/* Association and support limits */
#ifndef MAX_SUPPORTING_NODES
#define MAX_SUPPORTING_NODES 8
#endif

#ifndef MAX_ASSOCIATED_NODES
#define MAX_ASSOCIATED_NODES 16
#endif

/* Hash table and algorithmic constants */
#ifndef EDGE_HASH_SIZE
#define EDGE_HASH_SIZE 8192  /* Larger hash table */
#endif

#ifndef MAX_EDGE_CANDIDATES
#define MAX_EDGE_CANDIDATES 1000
#endif

#ifndef MAX_ATTENTION_HUBS
#define MAX_ATTENTION_HUBS 10
#endif

/* Processing and iteration limits */
#ifndef MAX_ITERATIONS_DEFAULT
#define MAX_ITERATIONS_DEFAULT 10
#endif

/* String length limits */
#ifndef MAX_DESCRIPTION_LEN
#define MAX_DESCRIPTION_LEN 256
#endif

#ifndef MAX_REASON_LEN
#define MAX_REASON_LEN 128
#endif

/* =============================================================================
 * ENHANCED ENUMERATIONS
 * =============================================================================
 */

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
    EDGE_BRIDGING,          /* long-range semantic bridge */
    
    /* Additional types from current code */
    EDGE_SEQUENCE,          /* Sequential relationship (token order) */
    EDGE_SEMANTIC           /* Semantic relationship */
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

/* =============================================================================
 * ENHANCED ATTENTION STRUCTURES
 * =============================================================================
 */

/* Multi-head attention structure for complex edge scoring */
typedef struct {
    float attention_weights[MAX_ATTENTION_HEADS][EMBEDDING_DIM];
    float attention_scores[MAX_ATTENTION_HEADS];
    float combined_score;
    int head_count;
} MultiHeadAttentionScore;

/* Main attention score structure with ALL fields used across modules */
typedef struct {
    float attention_score;                   /* Overall attention score */
    float identity_similarity;              /* Identity embedding similarity */
    float context_similarity;               /* Context embedding similarity */
    float temporal_similarity;              /* Temporal embedding similarity */
    float semantic_similarity;              /* Semantic similarity component */
    float token_similarity;                 /* Token matching component */
    float importance_factor;                /* Importance weighting factor */
    float combined_score;                   /* Combined final score */
    char reason[MAX_DESCRIPTION_LEN];       /* Explanation of the score */
} AttentionScore;

/* =============================================================================
 * COMPLETE GRAPH NODE STRUCTURE
 * =============================================================================
 */

/* Graph node with ALL fields actually used in the codebase */
typedef struct {
    /* Multi-scale embeddings - used throughout modules */
    float embedding[EMBEDDING_DIM];           /* current working embedding */
    float identity_embedding[EMBEDDING_DIM];  /* original/anchored identity */
    float context_embedding[EMBEDDING_DIM];   /* contextual refinements */
    float temporal_embedding[EMBEDDING_DIM];  /* temporal context evolution */
    
    /* Node identification and type */
    int node_id;                               /* Unique node identifier */
    NodeType type;                            /* ADDED: what kind of node this is */
    char token_text[MAX_TOKEN_LEN];           /* Original token text */
    char token[MAX_TOKEN_LEN];                /* ADDED: Alternative field name used in legacy */
    
    /* State and lifecycle */
    int active;                               /* Is this node active? */
    int access_count;                         /* ADDED: for memory management */
    float importance_score;                   /* Dynamic importance [0,1] */
    float attention_centrality;               /* ADDED: how much attention it attracts */
    int last_accessed_iteration;              /* Last time this node was used */
    
    /* Hierarchy and compression */
    int is_summary;                           /* Is this a summary node? */
    int is_global;                            /* Is this a global context node? */
    int compression_level;                    /* Level of compression (0=original) */
} GraphNode;

/* =============================================================================
 * COMPLETE GRAPH EDGE STRUCTURE
 * =============================================================================
 */

/* Graph edge with ALL fields actually used */
typedef struct {
    /* Basic edge properties */
    int src;                                  /* Source node index */
    int dst;                                  /* Destination node index */
    EdgeType type;                           /* Type of relationship */
    float weight;                            /* Edge strength [0,1] */
    float attention_score;                   /* Attention-based score */
    float activation;                        /* ADDED: Edge activation level */
    
    /* State and metadata */
    int active;                              /* Is this edge active? */
    int last_updated;                        /* Last iteration when updated */
    int creation_iteration;                  /* ADDED: when this edge was created */
    int usage_count;                         /* ADDED: how often it's been used */
    
    /* Advanced properties */
    int bidirectional;                       /* Is this edge bidirectional? */
    float confidence;                        /* Confidence in this relationship */
    float decay_factor;                      /* ADDED: for edge aging */
    AttentionScore detailed_attention;       /* ADDED: full attention breakdown */
} GraphEdge;

/* =============================================================================
 * COMPLETE MEMORY SLOT STRUCTURE
 * =============================================================================
 */

/* Memory slot with ALL fields used across modules */
typedef struct {
    /* Core memory content */
    float key[EMBEDDING_DIM];                /* Memory addressing key */
    float value[EMBEDDING_DIM];              /* Stored memory value */
    float context_key[EMBEDDING_DIM];        /* ADDED: contextual addressing key */
    char description[MAX_DESCRIPTION_LEN];   /* Human-readable description */
    
    /* Memory state */
    int active;                              /* Is this slot in use? */
    int in_use;                              /* ADDED: alternative field name */
    float importance;                        /* Memory importance score */
    float importance_score;                  /* ADDED: alternative field name */
    float access_count;                      /* ADDED: for LRU eviction */
    
    /* Timing and lifecycle */
    int last_accessed;                       /* Last access iteration */
    int access_count_int;                    /* ADDED: integer access count */
    int created_iteration;                   /* When this memory was created */
    int creation_time;                       /* ADDED: when this memory was formed */
    
    /* Associated graph nodes */
    int associated_nodes[MAX_ASSOCIATED_NODES]; /* Linked node indices */
    int associated_node_count;               /* Number of associated nodes */
    int node_count;                          /* ADDED: alternative field name */
} MemorySlot;

/* =============================================================================
 * COMPLETE SYMBOLIC FACT STRUCTURE
 * =============================================================================
 */

/* Symbolic fact with ALL fields used */
typedef struct {
    /* Core fact content */
    char subject[MAX_TOKEN_LEN];             /* Subject of the fact */
    char relation[MAX_TOKEN_LEN];            /* Relationship/predicate */
    char object[MAX_TOKEN_LEN];              /* Object of the fact */
    
    /* Fact metadata */
    float confidence;                        /* Confidence in this fact [0,1] */
    int derived;                             /* Was this fact derived (vs direct)? */
    int iteration_added;                     /* When this fact was added */
    int derivation_depth;                    /* ADDED: how many inference steps to derive */
    
    /* Supporting evidence */
    int supporting_nodes[MAX_SUPPORTING_NODES]; /* Nodes that support this fact */
    int support_count;                       /* Number of supporting nodes */
} SymbolicFact;

/* =============================================================================
 * COMPLETE VOCABULARY ENTRY STRUCTURE
 * =============================================================================
 */

/* Vocabulary entry with ALL fields used (including 19-year cache!) */
typedef struct {
    char token[MAX_TOKEN_LEN];               /* Token string */
    float embedding[EMBEDDING_DIM];          /* ADDED: token embedding from legacy */
    int frequency;                           /* Frequency of occurrence */
    int node_id;                            /* Associated node ID */
    float importance;                        /* Token importance score */
    time_t first_seen_time;                 /* RESTORED: When first seen (19-year eviction) */
    
    /* Additional fields from legacy modules */
    float context_variance;                  /* ADDED: how much context affects meaning */
    int contexts_seen;                       /* ADDED: number of different contexts */
} VocabEntry;

/* =============================================================================
 * CONFLICT TRACKING AND RESOLUTION STRUCTURES
 * =============================================================================
 */

/* Dynamic conflict description for detailed conflict tracking */
typedef struct {
    char* data;                              /* Dynamic text buffer */
    size_t size;                             /* Current size of text */
    size_t capacity;                         /* Allocated buffer capacity */
    size_t max_capacity;                     /* Maximum allowed capacity */
} DynamicConflictDescription;

/* Conflict tracking constants */
#ifndef CONFLICT_DESCRIPTION_INITIAL_SIZE
#define CONFLICT_DESCRIPTION_INITIAL_SIZE 256
#endif

#ifndef MAX_CONFLICT_DESCRIPTION_SIZE
#define MAX_CONFLICT_DESCRIPTION_SIZE 8192
#endif

#ifndef CONFLICT_DESCRIPTION_GROWTH_FACTOR
#define CONFLICT_DESCRIPTION_GROWTH_FACTOR 2
#endif

#ifndef CONFLICT_DESCRIPTION_SHRINK_THRESHOLD
#define CONFLICT_DESCRIPTION_SHRINK_THRESHOLD 4
#endif

/* Enhanced edge candidate for dynamic addition */
typedef struct {
    int src;                                 /* Source node */
    int dst;                                 /* Destination node */
    EdgeType type;                          /* Proposed edge type */
    EdgeType proposed_type;                 /* ADDED: alternative field name */
    float attention_score;                   /* Attention-based score */
    float similarity_score;                  /* ADDED: similarity component */
    float combined_score;                    /* ADDED: combined scoring */
    float confidence;                        /* ADDED: confidence in proposal */
    float score;                            /* ADDED: legacy score field */
    char reason[MAX_REASON_LEN];            /* Reason for this candidate */
    char reasoning[MAX_REASON_LEN];         /* ADDED: alternative field name */
} EdgeCandidate;

/* Edge hash table structures for deduplication */
typedef struct {
    int src;
    int dst;
    EdgeType type;
} EdgeKey;

typedef struct EdgeHashEntry {
    EdgeKey key;
    int exists;                              /* ADDED: existence flag */
    struct EdgeHashEntry* next;
} EdgeHashEntry;

/* =============================================================================
 * MAIN MODEL STRUCTURE WITH ALL FIELDS
 * =============================================================================
 */

/* Complete Galileo model structure with ALL fields used across modules */
typedef struct {
    /* Core graph components */
    GraphNode nodes[MAX_TOKENS];
    GraphEdge edges[MAX_EDGES];
    int num_nodes;
    int num_edges;
    
    /* Multi-model safe state */
    int symbolic_iteration_count;            /* ADDED: Was static - race condition fix */
    
    /* Edge deduplication hash table */
    EdgeHashEntry* edge_hash[EDGE_HASH_SIZE];
    
    /* Enhanced memory system */
    MemorySlot memory_slots[MAX_MEMORY_SLOTS];  /* RENAMED: memory module expects this field name */
    int num_memory_slots;
    
    /* Symbolic reasoning */
    SymbolicFact facts[MAX_FACTS];
    int num_facts;
    
    /* Global context and attention hubs */
    int global_node_idx;
    int attention_hubs[MAX_ATTENTION_HUBS];         
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
    EdgeCandidate edge_candidates[MAX_EDGE_CANDIDATES]; 
    int num_candidates;
    int max_edges_per_iteration;        
    
    /* Conflict and consistency tracking with dynamic buffers */
    DynamicConflictDescription** resolved_conflicts; /* Dynamic array of pointers to conflict descriptions */
    int num_resolved_conflicts;             /* Number of resolved conflicts */
    int resolved_conflicts_capacity;        /* Capacity of resolved_conflicts array */
    
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
 * MODULE INTERFACE TYPES
 * =============================================================================
 */

/* Generic module info structure for dynamic loading */
typedef struct {
    void* handle;                           /* dlopen handle */
    const char* name;                       /* Module name */
    int loaded;                             /* Is module loaded? */
    int required;                           /* Is module required? */
    int (*init_func)(void);                /* Module initialization function */
    void (*cleanup_func)(void);            /* Module cleanup function */
} ModuleInfo;

/* Performance tracking for module system - REMOVED DUPLICATE */

#endif /* GALILEO_TYPES_H */
