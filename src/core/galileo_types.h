/* =============================================================================
 * galileo/src/core/galileo_types.h - Fundamental Type Definitions
 * 
 * Core type definitions and structures for the Galileo v42 system.
 * This header defines all the fundamental data structures used throughout
 * the modular architecture.
 * 
 * This is the foundation that all other modules build upon - every structure,
 * enum, and constant that's shared across modules is defined here.
 * =============================================================================
 */

#ifndef GALILEO_TYPES_H
#define GALILEO_TYPES_H

#include <stdint.h>
#include <stddef.h>  /* For size_t */

/* =============================================================================
 * FUNDAMENTAL CONSTANTS
 * All constants can be overridden at compile time with -DCONSTANT_NAME=value
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
#define MAX_TOKENS 10000
#endif

#ifndef MAX_EDGES
#define MAX_EDGES 50000
#endif

/* Memory system limits */
#ifndef MAX_MEMORY_SLOTS
#define MAX_MEMORY_SLOTS 1000
#endif

/* Symbolic reasoning limits */
#ifndef MAX_FACTS
#define MAX_FACTS 5000
#endif

/* Vocabulary management */
#ifndef MAX_VOCAB_SIZE
#define MAX_VOCAB_SIZE 5000
#endif

/* Neural network dimensions */
#ifndef EMBEDDING_DIM
#define EMBEDDING_DIM 512
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
#define EDGE_HASH_SIZE 1024
#endif

#ifndef MAX_EDGE_CANDIDATES
#define MAX_EDGE_CANDIDATES 1000
#endif

#ifndef MAX_ATTENTION_HUBS
#define MAX_ATTENTION_HUBS 10
#endif

/* Processing and iteration limits */
#ifndef MAX_ITERATIONS_DEFAULT
#define MAX_ITERATIONS_DEFAULT 8
#endif

#ifndef MAX_EDGES_PER_ITERATION_DEFAULT
#define MAX_EDGES_PER_ITERATION_DEFAULT 10
#endif

/* Threshold defaults (can be overridden at runtime via model parameters) */
#ifndef SIMILARITY_THRESHOLD_DEFAULT
#define SIMILARITY_THRESHOLD_DEFAULT 0.85f
#endif

#ifndef ATTENTION_THRESHOLD_DEFAULT
#define ATTENTION_THRESHOLD_DEFAULT 0.75f
#endif

#ifndef COMPRESSION_THRESHOLD_DEFAULT
#define COMPRESSION_THRESHOLD_DEFAULT 0.9f
#endif

#ifndef IMPORTANCE_DECAY_DEFAULT
#define IMPORTANCE_DECAY_DEFAULT 0.95f
#endif

/* Buffer and string limits */
#ifndef MAX_DESCRIPTION_LEN
#define MAX_DESCRIPTION_LEN 256
#endif

#ifndef MAX_REASON_LEN
#define MAX_REASON_LEN 128
#endif

/* Dynamic buffer management for conflict descriptions */
#ifndef CONFLICT_DESCRIPTION_INITIAL_SIZE
#define CONFLICT_DESCRIPTION_INITIAL_SIZE 1024
#endif

#ifndef CONFLICT_DESCRIPTION_GROWTH_FACTOR
#define CONFLICT_DESCRIPTION_GROWTH_FACTOR 2
#endif

#ifndef CONFLICT_DESCRIPTION_SHRINK_THRESHOLD
#define CONFLICT_DESCRIPTION_SHRINK_THRESHOLD 4
#endif

/* Optional maximum size limit - if undefined, no limit */
#ifdef MAX_CONFLICT_DESCRIPTION_LIMIT
/* User can define this at compile time if they want a hard limit */
#endif

/* Dynamic conflict description structure */
typedef struct {
    char* data;                             /* Dynamic buffer */
    size_t size;                           /* Current buffer size */
    size_t capacity;                       /* Allocated capacity */
    size_t max_capacity;                   /* Maximum allowed capacity (if limited) */
} DynamicConflictDescription;

/* =============================================================================
 * ENUMERATIONS
 * =============================================================================
 */

/* Edge types for different kinds of relationships */
typedef enum {
    EDGE_SEQUENCE,      /* Sequential relationship (token order) */
    EDGE_SIMILARITY,    /* Semantic similarity relationship */
    EDGE_ATTENTION,     /* Attention-based relationship */
    EDGE_CAUSAL,        /* Causal relationship */
    EDGE_SEMANTIC,      /* Semantic relationship */
    EDGE_TEMPORAL,      /* Temporal relationship */
    EDGE_HIERARCHICAL,  /* Hierarchical relationship */
    EDGE_GLOBAL,        /* Global context relationship */
    EDGE_SUMMARY        /* Summary/compression relationship */
} EdgeType;

/* =============================================================================
 * CORE DATA STRUCTURES
 * =============================================================================
 */

/* Graph node representing a token, concept, or summary */
typedef struct {
    /* Core embeddings - multi-scale representation */
    float identity_embedding[EMBEDDING_DIM];    /* Core identity vector */
    float context_embedding[EMBEDDING_DIM];     /* Contextual information */
    float temporal_embedding[EMBEDDING_DIM];    /* Temporal/sequential info */
    
    /* Node metadata */
    int node_id;                               /* Unique node identifier */
    char token_text[MAX_TOKEN_LEN];           /* Original token text */
    int active;                               /* Is this node active? */
    float importance_score;                   /* Dynamic importance [0,1] */
    int last_accessed_iteration;              /* Last time this node was used */
    
    /* Node type and classification */
    int is_summary;                           /* Is this a summary node? */
    int is_global;                            /* Is this a global context node? */
    int compression_level;                    /* Level of compression (0=original) */
} GraphNode;

/* Graph edge connecting two nodes */
typedef struct {
    int src;                                  /* Source node index */
    int dst;                                  /* Destination node index */
    EdgeType type;                           /* Type of relationship */
    float weight;                            /* Edge strength [0,1] */
    float attention_score;                   /* Attention-based score */
    int active;                              /* Is this edge active? */
    int last_updated;                        /* Last iteration when updated */
    
    /* Edge metadata */
    int bidirectional;                       /* Is this edge bidirectional? */
    float confidence;                        /* Confidence in this relationship */
} GraphEdge;

/* Memory slot for contextual memory system */
typedef struct {
    float key[EMBEDDING_DIM];                /* Memory addressing key */
    float value[EMBEDDING_DIM];              /* Stored memory value */
    char description[MAX_DESCRIPTION_LEN];   /* Human-readable description */
    
    /* Memory metadata */
    int active;                              /* Is this slot in use? */
    float importance;                        /* Memory importance score */
    int last_accessed;                       /* Last access iteration */
    int access_count;                        /* Number of times accessed */
    int created_iteration;                   /* When this memory was created */
    
    /* Associated graph nodes */
    int associated_nodes[MAX_ASSOCIATED_NODES]; /* Linked node indices */
    int associated_node_count;               /* Number of associated nodes */
} MemorySlot;

/* Symbolic fact for logical reasoning */
typedef struct {
    char subject[MAX_TOKEN_LEN];             /* Subject of the fact */
    char relation[MAX_TOKEN_LEN];            /* Relationship/predicate */
    char object[MAX_TOKEN_LEN];              /* Object of the fact */
    
    /* Fact metadata */
    float confidence;                        /* Confidence in this fact [0,1] */
    int derived;                             /* Was this fact derived (vs direct)? */
    int iteration_added;                     /* When this fact was added */
    
    /* Supporting evidence */
    int supporting_nodes[MAX_SUPPORTING_NODES]; /* Nodes that support this fact */
    int support_count;                       /* Number of supporting nodes */
} SymbolicFact;

/* Vocabulary entry for efficient token management */
typedef struct {
    char token[MAX_TOKEN_LEN];               /* Token string */
    int frequency;                           /* Frequency of occurrence */
    int node_id;                            /* Associated node ID */
    float importance;                        /* Token importance score */
} VocabEntry;

/* Edge candidate for dynamic graph construction */
typedef struct {
    int src;                                 /* Source node */
    int dst;                                 /* Destination node */
    EdgeType type;                          /* Proposed edge type */
    float attention_score;                   /* Attention-based score */
    char reason[MAX_REASON_LEN];            /* Reason for this candidate */
} EdgeCandidate;

/* Attention score with detailed breakdown */
typedef struct {
    float attention_score;                   /* Overall attention score */
    float identity_similarity;              /* Identity embedding similarity */
    float context_similarity;               /* Context embedding similarity */
    float temporal_similarity;              /* Temporal embedding similarity */
    char reason[MAX_DESCRIPTION_LEN];       /* Explanation of the score */
} AttentionScore;

/* Edge hash table entry for deduplication */
typedef struct {
    int src;
    int dst;
    EdgeType type;
} EdgeKey;

typedef struct EdgeHashEntry {
    EdgeKey key;
    struct EdgeHashEntry* next;
} EdgeHashEntry;

/* =============================================================================
 * MAIN MODEL STRUCTURE
 * =============================================================================
 */

/* The complete Galileo model containing all subsystems */
typedef struct {
    /* Graph neural network components */
    GraphNode nodes[MAX_TOKENS];            /* All graph nodes */
    GraphEdge edges[MAX_EDGES];             /* All graph edges */
    int num_nodes;                          /* Current number of nodes */
    int num_edges;                          /* Current number of edges */
    
    /* Memory system */
    MemorySlot memory_slots[MAX_MEMORY_SLOTS]; /* Contextual memory */
    int num_memory_slots;                   /* Active memory slots */
    
    /* Symbolic reasoning system */
    SymbolicFact facts[MAX_FACTS];          /* Learned facts */
    int num_facts;                          /* Number of facts */
    
    /* Global context and special nodes */
    int global_node_idx;                    /* Index of global context node */
    int attention_hubs[MAX_ATTENTION_HUBS]; /* Important hub nodes */
    int num_attention_hubs;                 /* Number of hub nodes */
    
    /* Enhanced message passing arrays */
    float node_messages_local[MAX_TOKENS][EMBEDDING_DIM];     /* Local messages */
    float node_messages_global[MAX_TOKENS][EMBEDDING_DIM];    /* Global messages */
    float node_messages_attention[MAX_TOKENS][EMBEDDING_DIM]; /* Attention messages */
    float node_updates[MAX_TOKENS][EMBEDDING_DIM];           /* Node updates */
    
    /* Learning and adaptation parameters */
    float similarity_threshold;             /* Threshold for similarity edges */
    float attention_threshold;              /* Threshold for attention edges */
    float compression_threshold;            /* Threshold for compression */
    float importance_decay;                 /* Rate of importance decay */
    int max_iterations;                     /* Maximum processing iterations */
    int current_iteration;                  /* Current iteration number */
    
    /* Dynamic edge management */
    EdgeCandidate edge_candidates[MAX_EDGE_CANDIDATES]; /* Candidate edges for addition */
    int num_candidates;                     /* Number of edge candidates */
    int max_edges_per_iteration;           /* Max edges to add per iteration */
    
    /* Edge deduplication hash table */
    EdgeHashEntry* edge_hash[EDGE_HASH_SIZE]; /* Hash table for edge dedup */
    
    /* Conflict and consistency tracking with dynamic buffers */
    DynamicConflictDescription** resolved_conflicts; /* Dynamic array of pointers to conflict descriptions */
    int num_resolved_conflicts;             /* Number of resolved conflicts */
    int resolved_conflicts_capacity;        /* Capacity of resolved_conflicts array */
    
    /* Enhanced vocabulary system */
    VocabEntry vocabulary[MAX_VOCAB_SIZE];  /* Vocabulary management */
    int vocab_size;                         /* Current vocabulary size */
    
    /* Performance and efficiency tracking */
    int total_edges_added;                  /* Total edges added across all iterations */
    int total_compressions;                 /* Total compressions performed */
    int total_symbolic_calls;               /* Total symbolic inference calls */
    float avg_node_degree;                  /* Average node degree */
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

#endif /* GALILEO_TYPES_H */
