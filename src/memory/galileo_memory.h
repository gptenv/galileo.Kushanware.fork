/* =============================================================================
 * galileo/src/memory/galileo_memory.h - Memory Management Module Public API
 * 
 * Public header for the memory management module containing all functions
 * for contextual memory addressing, adaptive compression, hierarchical
 * summarization, and intelligent memory retrieval.
 * 
 * This module gives Galileo the ability to efficiently manage memory through
 * context-aware addressing, smart compression, and importance-based eviction.
 * =============================================================================
 */

#ifndef GALILEO_MEMORY_H
#define GALILEO_MEMORY_H

#include "../core/galileo_types.h"
#include <stddef.h>

/* =============================================================================
 * MEMORY WRITE OPERATIONS
 * =============================================================================
 */

/* Enhanced memory write with contextual addressing and metadata */
void galileo_enhanced_memory_write(GalileoModel* model, 
                                  const float* key, 
                                  const float* value, 
                                  const char* description, 
                                  int* associated_nodes, 
                                  int node_count);

/* =============================================================================
 * MEMORY READ AND RETRIEVAL OPERATIONS
 * =============================================================================
 */

/* Contextual memory read with intelligent retrieval and blending */
void galileo_contextual_memory_read(GalileoModel* model, 
                                   const float* query, 
                                   const float* context, 
                                   float* result);

/* =============================================================================
 * COMPRESSION AND SUMMARIZATION
 * =============================================================================
 */

/* Adaptive compression based on importance and recency */
void galileo_adaptive_compression(GalileoModel* model);

/* Create summary node from multiple source nodes */
int galileo_create_summary_node(GalileoModel* model, 
                               int* source_nodes, 
                               int count);

/* =============================================================================
 * MEMORY ANALYSIS AND UTILITIES
 * =============================================================================
 */

/* Memory importance scoring and analysis */
#ifdef GALILEO_EXPOSE_INTERNAL_MEMORY
float compute_memory_importance(const MemorySlot* slot, int current_iteration);
int find_lru_slot(GalileoModel* model);
#endif

/* Memory addressing utilities */
#ifdef GALILEO_EXPOSE_INTERNAL_MEMORY
void generate_contextual_key(const float* query, const float* context, float* result);
float cosine_similarity(const float* a, const float* b, int dim);
#endif

/* =============================================================================
 * CONFIGURATION CONSTANTS
 * All constants can be overridden at compile time
 * =============================================================================
 */

/* Memory retrieval parameters */
#ifndef MAX_MEMORY_MATCHES
#define MAX_MEMORY_MATCHES 3
#endif

#ifndef MIN_MEMORY_SIMILARITY
#define MIN_MEMORY_SIMILARITY 0.1f
#endif

/* Contextual key generation weights */
#ifndef CONTEXTUAL_QUERY_WEIGHT
#define CONTEXTUAL_QUERY_WEIGHT 0.7f
#endif

#ifndef CONTEXTUAL_CONTEXT_WEIGHT
#define CONTEXTUAL_CONTEXT_WEIGHT 0.3f
#endif

/* Memory importance calculation factors */
#ifndef MEMORY_RECENCY_DECAY_FACTOR
#define MEMORY_RECENCY_DECAY_FACTOR 10.0f
#endif

#ifndef MEMORY_FREQUENCY_BOOST_FACTOR
#define MEMORY_FREQUENCY_BOOST_FACTOR 10.0f
#endif

#ifndef MEMORY_NODE_IMPORTANCE_FACTOR
#define MEMORY_NODE_IMPORTANCE_FACTOR 10.0f
#endif

/* Compression parameters */
#ifndef MIN_COMPRESSION_AGE
#define MIN_COMPRESSION_AGE 3
#endif

#ifndef MAX_COMPRESSIONS_PER_ITERATION
#define MAX_COMPRESSIONS_PER_ITERATION 3
#endif

#ifndef MIN_COMPRESSION_GROUP_SIZE
#define MIN_COMPRESSION_GROUP_SIZE 2
#endif

#ifndef MAX_COMPRESSION_GROUP_SIZE
#define MAX_COMPRESSION_GROUP_SIZE 4
#endif

/* Compression scoring thresholds */
#ifndef MIN_COMPRESSION_IMPORTANCE
#define MIN_COMPRESSION_IMPORTANCE 0.2f
#endif

#ifndef COMPRESSION_AGE_FACTOR_DIVISOR
#define COMPRESSION_AGE_FACTOR_DIVISOR 5.0f
#endif

/* Summary node creation parameters */
#ifndef MAX_SUMMARY_SOURCE_NODES
#define MAX_SUMMARY_SOURCE_NODES 10
#endif

#ifndef MIN_SUMMARY_SOURCE_NODES
#define MIN_SUMMARY_SOURCE_NODES 1
#endif

/* Memory eviction and LRU parameters */
#ifndef MEMORY_ACCESS_COUNT_BOOST
#define MEMORY_ACCESS_COUNT_BOOST 1.0f
#endif

#ifndef MEMORY_IMPORTANCE_THRESHOLD
#define MEMORY_IMPORTANCE_THRESHOLD 0.1f
#endif

/* Default importance values */
#ifndef DEFAULT_MEMORY_IMPORTANCE
#define DEFAULT_MEMORY_IMPORTANCE 0.5f
#endif

#ifndef MEMORY_IMPORTANCE_NODE_BOOST
#define MEMORY_IMPORTANCE_NODE_BOOST 0.1f
#endif

/* =============================================================================
 * MEMORY STATISTICS AND ANALYSIS STRUCTURES
 * =============================================================================
 */

/* Memory usage statistics */
typedef struct {
    int active_slots;                       /* Number of active memory slots */
    int total_slots;                        /* Total available memory slots */
    float utilization_percentage;           /* Memory utilization as percentage */
    float average_importance;               /* Average importance of active memories */
    float average_access_count;             /* Average access count */
    int oldest_memory_age;                  /* Age of oldest memory */
    int newest_memory_age;                  /* Age of newest memory */
    int total_evictions;                    /* Total number of evictions performed */
} MemoryStatistics;

/* Memory match result for retrieval operations */
typedef struct {
    int slot_index;                         /* Index of matching memory slot */
    float similarity;                       /* Similarity score to query */
    float importance;                       /* Importance score of memory */
    float combined_score;                   /* Combined retrieval score */
    char description[MAX_DESCRIPTION_LEN];  /* Memory description */
} MemoryMatch;

/* =============================================================================
 * ADVANCED MEMORY OPERATIONS
 * =============================================================================
 */

/* Get comprehensive memory statistics */
#ifdef GALILEO_EXPOSE_INTERNAL_MEMORY
void get_memory_statistics(GalileoModel* model, MemoryStatistics* stats);
#endif

/* Find best memory matches for a query */
#ifdef GALILEO_EXPOSE_INTERNAL_MEMORY
int find_memory_matches(GalileoModel* model, 
                       const float* contextual_query, 
                       MemoryMatch* matches, 
                       int max_matches);
#endif

/* Memory slot validation and cleanup */
#ifdef GALILEO_EXPOSE_INTERNAL_MEMORY
int validate_memory_slot(const MemorySlot* slot);
void cleanup_inactive_memory_slots(GalileoModel* model);
#endif

/* =============================================================================
 * COMPRESSION ANALYSIS STRUCTURES
 * =============================================================================
 */

/* Compression candidate analysis */
typedef struct {
    int node_index;                         /* Index of node being analyzed */
    float compression_score;                /* Overall compression score */
    float importance;                       /* Node importance */
    int last_access_age;                    /* Age since last access */
    int connectivity;                       /* Number of edges */
    int is_candidate;                       /* Is this a good compression candidate? */
} CompressionCandidate;

/* Compression operation result */
typedef struct {
    int groups_compressed;                  /* Number of node groups compressed */
    int nodes_compressed;                   /* Total nodes compressed */
    int summary_nodes_created;             /* Number of summary nodes created */
    float compression_ratio;                /* Compression ratio achieved */
    int iteration_performed;               /* Iteration when compression happened */
} CompressionResult;

/* =============================================================================
 * ADVANCED COMPRESSION OPERATIONS
 * =============================================================================
 */

/* Analyze compression candidates */
#ifdef GALILEO_EXPOSE_INTERNAL_MEMORY
int analyze_compression_candidates(GalileoModel* model, 
                                  CompressionCandidate* candidates, 
                                  int max_candidates);
#endif

/* Perform targeted compression on specific nodes */
#ifdef GALILEO_EXPOSE_INTERNAL_MEMORY
CompressionResult compress_node_group(GalileoModel* model, 
                                     int* node_indices, 
                                     int count);
#endif

/* =============================================================================
 * MODULE INTERFACE
 * =============================================================================
 */

/* Module info for dynamic loading */
typedef struct {
    const char* name;
    const char* version;
    int (*init_func)(void);
    void (*cleanup_func)(void);
} MemoryModuleInfo;

extern MemoryModuleInfo memory_module_info;

#endif /* GALILEO_MEMORY_H */
