/* =============================================================================
 * galileo/src/core/galileo_core.c - Complete Core Module Implementation
 * 
 * ENHANCED VERSION - All improvements retained + first_seen_time restored
 * Hot-loadable shared library implementing the core Galileo model structure,
 * lifecycle management, and all basic operations with maximum robustness.
 * 
 * 🎯 FEATURES:
 * - Full lazy loading support with enhanced module coordination
 * - 19-year cache eviction strategy support (first_seen_time field)
 * - Race condition fixes and enhanced thread safety
 * - Memory leak prevention and smart resource management
 * - Enhanced attention mechanisms with NaN protection
 * - Optimized O(n²) → O(n log n) graph operations
 * - Comprehensive error checking and bounds validation
 * - Performance monitoring and debugging capabilities
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

#include "galileo_core.h"
#include "galileo_module_loader.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <dlfcn.h>
#include <unistd.h>

/* Thread support - portable across UNIX systems */
#if defined(_POSIX_THREADS) || defined(__APPLE__) || defined(__linux__)
    #include <pthread.h>
    #define GALILEO_HAS_PTHREAD 1
#else
    #define GALILEO_HAS_PTHREAD 0
#endif

/* =============================================================================
 * MODULE METADATA AND INITIALIZATION
 * =============================================================================
 */

static int core_module_initialized = 0;
static int active_model_count = 0;
static ModulePerformanceStats g_performance_stats = {0};

#if GALILEO_HAS_PTHREAD
static pthread_mutex_t core_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Thread-safe model counting */
static void increment_model_count_safe(void) {
    pthread_mutex_lock(&core_mutex);
    active_model_count++;
    pthread_mutex_unlock(&core_mutex);
}

static void decrement_model_count_safe(void) {
    pthread_mutex_lock(&core_mutex);
    active_model_count--;
    pthread_mutex_unlock(&core_mutex);
}
#else
/* Non-threaded fallbacks */
static void increment_model_count_safe(void) { active_model_count++; }
static void decrement_model_count_safe(void) { active_model_count--; }
#endif

/* Module initialization with enhanced logging */
static int core_module_init(void) {
    if (core_module_initialized) {
        return 0;  /* Already initialized */
    }
    
    fprintf(stderr, "🧠 Core module v42.1 initializing...\n");
    fprintf(stderr, "   ✅ Enhanced robustness and error checking\n");
    fprintf(stderr, "   ✅ 19-year cache eviction strategy support\n");
    fprintf(stderr, "   ✅ Race condition fixes and thread safety\n");
    fprintf(stderr, "   ✅ Memory leak prevention\n");
    fprintf(stderr, "   ✅ Optimized graph operations\n");
    
    /* Initialize core subsystems */
    active_model_count = 0;
    memset(&g_performance_stats, 0, sizeof(g_performance_stats));
    
    /* Seed random number generator for embedding initialization */
    srand((unsigned int)time(NULL));
    
    core_module_initialized = 1;
    fprintf(stderr, "✅ Core module ready! Multi-model support enabled.\n");
    return 0;
}

/* Enhanced module cleanup with detailed logging */
static void core_module_cleanup(void) {
    if (!core_module_initialized) {
        return;
    }
    
    fprintf(stderr, "🧠 Core module shutting down...\n");
    
    if (active_model_count > 0) {
        fprintf(stderr, "⚠️  Warning: %d models still active during shutdown\n", active_model_count);
    }
    
    /* Report final performance statistics */
    if (g_performance_stats.module_function_calls > 0) {
        fprintf(stderr, "📊 Final stats: %d function calls, most used: %s\n",
                g_performance_stats.module_function_calls,
                g_performance_stats.most_used_module[0] ? g_performance_stats.most_used_module : "none");
    }
    
    core_module_initialized = 0;
    fprintf(stderr, "✅ Core module cleanup complete.\n");
}

/* Enhanced module info structure for dynamic loading */
CoreModuleInfo core_module_info = {
    .name = "core",
    .version = "42.1.0-enhanced", 
    .init_func = core_module_init,
    .cleanup_func = core_module_cleanup
};

/* =============================================================================
 * ENHANCED UTILITY FUNCTIONS WITH VALIDATION
 * =============================================================================
 */

/* Safe string operations with enhanced validation */
void safe_strcpy(char* dest, const char* src, size_t dest_size) {
    if (!dest || !src || dest_size == 0) {
        return;  /* Graceful handling of invalid inputs */
    }
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';
}

/* Enhanced random number generation */
float random_float(void) {
    return (float)rand() / (float)RAND_MAX;
}

/* Random float in range [-1, 1] with better distribution */
float random_float_range(void) {
    return 2.0f * random_float() - 1.0f;
}

/* Enhanced vector operations with NaN protection */
float vector_dot_product(const float* a, const float* b, int dim) {
    if (!a || !b || dim <= 0) return 0.0f;
    
    double sum = 0.0;  /* Use double for better precision */
    for (int i = 0; i < dim; i++) {
        if (isfinite(a[i]) && isfinite(b[i])) {
            sum += (double)a[i] * (double)b[i];
        }
    }
    return isfinite(sum) ? (float)sum : 0.0f;
}

float vector_magnitude(const float* vec, int dim) {
    if (!vec || dim <= 0) return 0.0f;
    
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        if (isfinite(vec[i])) {
            sum += (double)vec[i] * (double)vec[i];
        }
    }
    return isfinite(sum) ? (float)sqrt(sum) : 0.0f;
}

float cosine_similarity(const float* a, const float* b, int dim) {
    if (!a || !b || dim <= 0) return 0.0f;
    
    float dot = vector_dot_product(a, b, dim);
    float mag_a = vector_magnitude(a, dim);
    float mag_b = vector_magnitude(b, dim);
    
    if (mag_a < 1e-8f || mag_b < 1e-8f) return 0.0f;
    
    float similarity = dot / (mag_a * mag_b);
    return isfinite(similarity) ? similarity : 0.0f;
}

void vector_normalize(float* vec, int dim) {
    if (!vec || dim <= 0) return;
    
    float mag = vector_magnitude(vec, dim);
    if (mag > 1e-8f) {
        for (int i = 0; i < dim; i++) {
            vec[i] /= mag;
        }
    }
}

void vector_add_scaled(float* dest, const float* src, float scale, int dim) {
    if (!dest || !src || dim <= 0 || !isfinite(scale)) return;
    
    for (int i = 0; i < dim; i++) {
        if (isfinite(src[i])) {
            dest[i] += scale * src[i];
        }
    }
}

void vector_copy(float* dest, const float* src, int dim) {
    if (!dest || !src || dim <= 0) return;
    memcpy(dest, src, dim * sizeof(float));
}

void vector_zero(float* vec, int dim) {
    if (!vec || dim <= 0) return;
    memset(vec, 0, dim * sizeof(float));
}

/* Enhanced hash function for token processing */
uint32_t enhanced_hash(const char* str) {
    if (!str) return 0;
    
    uint32_t hash = 5381;
    for (const char* c = str; *c; c++) {
        hash = ((hash << 5) + hash) + (unsigned char)*c;
    }
    return hash;
}

/* =============================================================================
 * DYNAMIC MODULE FUNCTION CALLING HELPERS
 * =============================================================================
 */

/* Get a function pointer from a loaded module */
void* get_module_function(const char* module_name, const char* function_name) {
    /* Check if module is loaded */
    if (!galileo_is_module_loaded(module_name)) {
        return NULL;
    }
    
    /* Get the module handle directly using dlopen with RTLD_NOLOAD */
    char module_path[512];
    snprintf(module_path, sizeof(module_path), "./build/lib/galileo/libgalileo_%s.so", module_name);
    
    void* module_handle = dlopen(module_path, RTLD_LAZY | RTLD_NOLOAD);
    if (!module_handle) {
        /* Try with absolute path construction */
        char* cwd = getcwd(NULL, 0);
        if (cwd) {
            snprintf(module_path, sizeof(module_path), "%s/build/lib/galileo/libgalileo_%s.so", cwd, module_name);
            module_handle = dlopen(module_path, RTLD_LAZY | RTLD_NOLOAD);
            free(cwd);
        }
        
        if (!module_handle) {
            fprintf(stderr, "⚠️  Could not get handle for module '%s': %s\n", 
                    module_name, dlerror());
            return NULL;
        }
    }
    
    /* Get the function pointer */
    void* func_ptr = dlsym(module_handle, function_name);
    if (!func_ptr) {
        fprintf(stderr, "⚠️  Function '%s' not found in module '%s': %s\n", 
                function_name, module_name, dlerror());
        return NULL;
    }
    
    /* Update performance stats */
    g_performance_stats.module_function_calls++;
    safe_strcpy(g_performance_stats.most_used_module, module_name, sizeof(g_performance_stats.most_used_module));
    
    return func_ptr;
}

/* =============================================================================
 * LAZY LOADING MODULE FUNCTION WRAPPERS
 * =============================================================================
 */

/* Safe wrapper to call graph module functions with lazy loading */
int call_graph_function_lazy(const char* function_name, GalileoModel* model) {
    /* Try to ensure graph module is loaded */
    if (ensure_module_loaded("graph") != 0) {
        return 0;  /* Module not available */
    }
    
    typedef void (*graph_func_t)(GalileoModel*);
    graph_func_t func = (graph_func_t)get_module_function("graph", function_name);
    
    if (func) {
        func(model);
        return 1;  /* Success */
    }
    
    return 0;  /* Function not available */
}

/* Safe wrapper to call symbolic module functions with lazy loading */
int call_symbolic_function_lazy(const char* function_name, GalileoModel* model) {
    /* Try to ensure symbolic module is loaded */
    if (ensure_module_loaded("symbolic") != 0) {
        return 0;  /* Module not available */
    }
    
    typedef void (*symbolic_func_t)(GalileoModel*);
    symbolic_func_t func = (symbolic_func_t)get_module_function("symbolic", function_name);
    
    if (func) {
        func(model);
        return 1;  /* Success */
    }
    
    return 0;  /* Function not available */
}

/* Safe wrapper to call memory module functions with lazy loading */
int call_memory_function_lazy(const char* function_name, GalileoModel* model) {
    /* Try to ensure memory module is loaded */
    if (ensure_module_loaded("memory") != 0) {
        return 0;  /* Module not available */
    }
    
    typedef void (*memory_func_t)(GalileoModel*);
    memory_func_t func = (memory_func_t)get_module_function("memory", function_name);
    
    if (func) {
        func(model);
        return 1;  /* Success */
    }
    
    return 0;  /* Function not available */
}

/* Safe wrapper to call heuristic module functions with lazy loading */
int call_heuristic_function_lazy(const char* function_name, GalileoModel* model, 
                                char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    /* Try to ensure heuristic module is loaded */
    if (ensure_module_loaded("heuristic") != 0) {
        return 0;  /* Module not available */
    }
    
    typedef int (*heuristic_func_t)(GalileoModel*, char[][MAX_TOKEN_LEN], int);
    heuristic_func_t func = (heuristic_func_t)get_module_function("heuristic", function_name);
    
    if (func) {
        return func(model, tokens, num_tokens);
    }
    
    return 0;  /* Function not available */
}

/* Safe wrapper to call utils module functions with lazy loading */
int call_utils_function_lazy(const char* function_name, GalileoModel* model) {
    /* Try to ensure utils module is loaded */
    if (ensure_module_loaded("utils") != 0) {
        return 0;  /* Module not available */
    }
    
    typedef void (*utils_func_t)(GalileoModel*);
    utils_func_t func = (utils_func_t)get_module_function("utils", function_name);
    
    if (func) {
        func(model);
        return 1;  /* Success */
    }
    
    return 0;  /* Function not available */
}

/* =============================================================================
 * MODULE AVAILABILITY CHECKING
 * =============================================================================
 */

/* Check if specific modules are available for use */
int galileo_has_graph_module(void) {
    return galileo_is_module_loaded("graph") || (ensure_module_loaded("graph") == 0);
}

int galileo_has_symbolic_module(void) {
    return galileo_is_module_loaded("symbolic") || (ensure_module_loaded("symbolic") == 0);
}

int galileo_has_memory_module(void) {
    return galileo_is_module_loaded("memory") || (ensure_module_loaded("memory") == 0);
}

int galileo_has_heuristic_module(void) {
    return galileo_is_module_loaded("heuristic") || (ensure_module_loaded("heuristic") == 0);
}

int galileo_has_utils_module(void) {
    return galileo_is_module_loaded("utils") || (ensure_module_loaded("utils") == 0);
}

/* =============================================================================
 * GALILEO MODEL LIFECYCLE MANAGEMENT
 * =============================================================================
 */

/* Enhanced model initialization with comprehensive setup */
GalileoModel* galileo_init(void) {
    if (!core_module_initialized) {
        fprintf(stderr, "❌ Error: Core module not initialized\n");
        return NULL;
    }
    
    printf("🚀 Galileo v42 Enhanced Edition initializing...\n");
    printf("   ✅ Race condition fixes and enhanced robustness\n");
    printf("   ✅ Memory leak fixes and smart resource management\n");
    printf("   ✅ Enhanced attention mechanisms with NaN protection\n");
    printf("   ✅ Optimized O(n²) → O(n log n) graph operations\n");
    printf("   ✅ Lazy module loading with hot-pluggable architecture\n");
    printf("   ✅ 19-year cache eviction strategy support\n");
    
    GalileoModel* model = malloc(sizeof(GalileoModel));
    if (!model) {
        fprintf(stderr, "❌ Failed to allocate memory for Galileo model\n");
        return NULL;
    }
    
    /* Initialize all fields to zero first - CRITICAL for safety */
    memset(model, 0, sizeof(GalileoModel));
    
    /* Initialize key parameters with enhanced defaults */
    model->similarity_threshold = 0.4f;
    model->attention_threshold = 0.3f;
    model->compression_threshold = 0.7f;
    model->importance_decay = 0.95f;
    model->max_iterations = 10;
    model->max_edges_per_iteration = 15;
    model->current_iteration = 0;
    
    /* Initialize global context node with enhanced setup */
    model->global_node_idx = 0;
    GraphNode* global_node = &model->nodes[0];
    global_node->node_id = 0;
    global_node->active = 1;
    global_node->is_global = 1;
    global_node->is_summary = 0;
    global_node->importance_score = 1.0f;
    global_node->last_accessed_iteration = 0;
    global_node->compression_level = 0;
    safe_strcpy(global_node->token_text, "<GLOBAL>", MAX_TOKEN_LEN);
    
    /* Initialize global node embeddings with enhanced random distribution */
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        global_node->identity_embedding[i] = random_float_range() * 0.1f;
        global_node->context_embedding[i] = random_float_range() * 0.1f;
        global_node->temporal_embedding[i] = random_float_range() * 0.1f;
    }
    
    model->num_nodes = 1;  /* Start with global node */
    
    /* Initialize attention hubs system */
    model->attention_hubs[0] = 0;  /* Global node is first hub */
    model->num_attention_hubs = 1;
    
    /* Initialize edge hash table for deduplication */
    for (int i = 0; i < EDGE_HASH_SIZE; i++) {
        model->edge_hash[i] = NULL;
    }
    
    increment_model_count_safe();
    printf("🎯 Enhanced model initialized with ID #%d (Model #%d)\n", 
           model->global_node_idx, active_model_count);
    printf("   Parameters: sim_thresh=%.2f, att_thresh=%.2f, max_iter=%d\n",
           model->similarity_threshold, model->attention_threshold, model->max_iterations);
    
    return model;
}

/* Enhanced model destruction with proper cleanup */
void galileo_destroy(GalileoModel* model) {
    if (!model) return;
    
    printf("🔥 Destroying Enhanced Galileo model...\n");
    printf("📊 Final stats: %d nodes, %d edges, %d facts, %d compressions\n",
           model->num_nodes, model->num_edges, model->num_facts, model->total_compressions);
    
    /* Clean up edge hash table to prevent memory leaks */
    for (int i = 0; i < EDGE_HASH_SIZE; i++) {
        EdgeHashEntry* entry = model->edge_hash[i];
        while (entry) {
            EdgeHashEntry* next = entry->next;
            free(entry);
            entry = next;
        }
        model->edge_hash[i] = NULL;
    }
    
    /* Report vocabulary statistics including 19-year cache info */
    if (model->vocab_size > 0) {
        time_t current_time = time(NULL);
        int old_entries = 0;
        for (int i = 0; i < model->vocab_size; i++) {
            double age_years = difftime(current_time, model->vocabulary[i].first_seen_time) / (365.25 * 24 * 3600);
            if (age_years > 19.0) {
                old_entries++;
            }
        }
        printf("📚 Vocabulary: %d entries, %d eligible for 19-year eviction\n", 
               model->vocab_size, old_entries);
    }
    
    /* Clear any sensitive data before freeing */
    memset(model, 0, sizeof(GalileoModel));
    free(model);
    
    decrement_model_count_safe();
    printf("✅ Enhanced model destroyed safely. (%d models remaining)\n", active_model_count);
}

/* =============================================================================
 * ENHANCED TOKEN AND NODE OPERATIONS
 * =============================================================================
 */

/* Enhanced token addition with comprehensive validation and 19-year tracking */
int galileo_add_token(GalileoModel* model, const char* token_text) {
    if (!model || !token_text || !core_module_initialized) {
        return -1;  /* Enhanced validation */
    }
    
    /* Validate token text */
    if (strlen(token_text) == 0 || strlen(token_text) >= MAX_TOKEN_LEN) {
        return -1;  /* Invalid token length */
    }
    
    /* Check capacity */
    if (model->num_nodes >= MAX_TOKENS) {
        fprintf(stderr, "⚠️  Warning: Maximum tokens reached (%d)\n", MAX_TOKENS);
        return -1;
    }
    
    /* Check if token already exists with enhanced search */
    for (int i = 1; i < model->num_nodes; i++) {  /* Skip global node */
        if (model->nodes[i].active && 
            !model->nodes[i].is_summary && 
            !model->nodes[i].is_global &&
            strcmp(model->nodes[i].token_text, token_text) == 0) {
            /* Update access time for existing token */
            model->nodes[i].last_accessed_iteration = model->current_iteration;
            return i;  /* Token already exists */
        }
    }
    
    /* Create new token node with enhanced initialization */
    int node_idx = model->num_nodes++;
    GraphNode* node = &model->nodes[node_idx];
    
    /* Zero out the node first for safety */
    memset(node, 0, sizeof(GraphNode));
    
    /* Initialize node properties */
    safe_strcpy(node->token_text, token_text, MAX_TOKEN_LEN);
    node->node_id = node_idx;
    node->active = 1;
    node->is_summary = 0;
    node->is_global = 0;
    node->importance_score = 0.5f;
    node->last_accessed_iteration = model->current_iteration;
    node->compression_level = 0;
    node->attention_centrality = 0.0f;
    
    /* Initialize embeddings with enhanced random distribution */
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        float base_val = random_float_range() * 0.1f;
        node->identity_embedding[i] = base_val;
        node->context_embedding[i] = base_val * 0.8f;  /* Correlated but different */
        node->temporal_embedding[i] = base_val * 0.6f;
    }
    
    /* Enhanced vocabulary management with 19-year tracking */
    int vocab_found = 0;
    for (int i = 0; i < model->vocab_size; i++) {
        if (strcmp(model->vocabulary[i].token, token_text) == 0) {
            model->vocabulary[i].frequency++;
            model->vocabulary[i].node_id = node_idx;  /* Update to latest node */
            vocab_found = 1;
            break;
        }
    }
    
    if (!vocab_found && model->vocab_size < MAX_VOCAB_SIZE) {
        VocabEntry* entry = &model->vocabulary[model->vocab_size];
        safe_strcpy(entry->token, token_text, MAX_TOKEN_LEN);
        entry->frequency = 1;
        entry->node_id = node_idx;
        entry->importance = 0.5f;
        entry->first_seen_time = time(NULL);  /* RESTORED: For 19-year cache eviction strategy */
        model->vocab_size++;
    } else if (!vocab_found) {
        fprintf(stderr, "⚠️  Warning: Vocabulary full, cannot add '%s'\n", token_text);
    }
    
    return node_idx;
}
/* Enhanced token embedding generation with improved caching */
float* get_enhanced_token_embedding(GalileoModel* model, const char* token_text, 
                                   int context_position) {
    if (!model || !token_text || !core_module_initialized) {
        return NULL;
    }
    
    /* Validate context position */
    if (context_position < 0 || context_position >= MAX_TOKENS) {
        return NULL;
    }
    
    /* Find the token node with enhanced search */
    int token_idx = -1;
    for (int i = 1; i < model->num_nodes; i++) {  /* Skip global node */
        if (model->nodes[i].active &&
            !model->nodes[i].is_summary && 
            !model->nodes[i].is_global &&
            strcmp(model->nodes[i].token_text, token_text) == 0) {
            token_idx = i;
            break;
        }
    }
    
    if (token_idx == -1) {
        return NULL;  /* Token not found */
    }
    
    /* Update access tracking */
    model->nodes[token_idx].last_accessed_iteration = model->current_iteration;
    
    /* Return the identity embedding (most stable representation) */
    return model->nodes[token_idx].identity_embedding;
}

/* =============================================================================
 * ENHANCED MODEL ANALYSIS AND STATISTICS
 * =============================================================================
 */

/* Enhanced graph statistics computation with detailed reporting */
void galileo_compute_graph_stats(GalileoModel* model) {
    if (!model || !core_module_initialized) {
        return;
    }
    
    /* Count node types with enhanced classification */
    int token_nodes = 0, summary_nodes = 0, global_nodes = 0, inactive_nodes = 0;
    float total_importance = 0.0f;
    int recent_nodes = 0;
    
    for (int i = 0; i < model->num_nodes; i++) {
        GraphNode* node = &model->nodes[i];
        
        if (!node->active) {
            inactive_nodes++;
            continue;
        }
        
        if (node->is_global) {
            global_nodes++;
        } else if (node->is_summary) {
            summary_nodes++;
        } else {
            token_nodes++;
        }
        
        total_importance += node->importance_score;
        
        /* Count recently accessed nodes */
        if (model->current_iteration - node->last_accessed_iteration <= 3) {
            recent_nodes++;
        }
    }
    
    /* Count edge types with enhanced categorization */
    int similarity_edges = 0, attention_edges = 0, semantic_edges = 0;
    int sequence_edges = 0, global_edges = 0, other_edges = 0;
    float total_edge_weight = 0.0f;
    
    for (int i = 0; i < model->num_edges; i++) {
        GraphEdge* edge = &model->edges[i];
        
        switch (edge->type) {
            case EDGE_SIMILARITY: similarity_edges++; break;
            case EDGE_ATTENTION: attention_edges++; break;
            case EDGE_SEMANTIC: semantic_edges++; break;
            case EDGE_SEQUENCE: sequence_edges++; break;
            case EDGE_GLOBAL: global_edges++; break;
            default: other_edges++; break;
        }
        
        total_edge_weight += edge->weight;
    }
    
    /* Calculate enhanced metrics */
    model->avg_node_degree = model->num_nodes > 0 ?
        (float)(model->num_edges * 2) / model->num_nodes : 0.0f;
    
    float avg_importance = model->num_nodes > 0 ? total_importance / model->num_nodes : 0.0f;
    float avg_edge_weight = model->num_edges > 0 ? total_edge_weight / model->num_edges : 0.0f;
    
    /* Display comprehensive statistics */
    printf("\n📊 === Enhanced Graph Statistics ===\n");
    printf("Nodes: %d total (%d active, %d inactive)\n", 
           model->num_nodes, model->num_nodes - inactive_nodes, inactive_nodes);
    printf("  - %d tokens, %d summaries, %d globals\n", token_nodes, summary_nodes, global_nodes);
    printf("  - %d recently accessed (last 3 iterations)\n", recent_nodes);
    printf("  - Average importance: %.3f\n", avg_importance);
    
    printf("Edges: %d total, avg degree: %.2f\n", model->num_edges, model->avg_node_degree);
    printf("  - %d sequence, %d similarity, %d attention\n", sequence_edges, similarity_edges, attention_edges);
    printf("  - %d semantic, %d global, %d other\n", semantic_edges, global_edges, other_edges);
    printf("  - Average weight: %.3f\n", avg_edge_weight);
    
    printf("Memory: %d/%d slots, Facts: %d/%d\n", 
           model->num_memory_slots, MAX_MEMORY_SLOTS, model->num_facts, MAX_FACTS);
    printf("Vocabulary: %d/%d entries\n", model->vocab_size, MAX_VOCAB_SIZE);
    
    printf("Performance: %d edges added, %d compressions, %d symbolic calls\n",
           model->total_edges_added, model->total_compressions, model->total_symbolic_calls);
    
    /* Report on 19-year cache status */
    if (model->vocab_size > 0) {
        time_t current_time = time(NULL);
        int old_entries = 0;
        for (int i = 0; i < model->vocab_size; i++) {
            double age_years = difftime(current_time, model->vocabulary[i].first_seen_time) / (365.25 * 24 * 3600);
            if (age_years > 19.0) {
                old_entries++;
            }
        }
        printf("Cache: %d entries eligible for 19-year eviction\n", old_entries);
    }
}

/* Update importance scores for all nodes */
void galileo_update_importance_scores(GalileoModel* model) {
    if (!model || !core_module_initialized) {
        return;
    }
    
    for (int i = 0; i < model->num_nodes; i++) {
        GraphNode* node = &model->nodes[i];
        
        /* Skip inactive nodes */
        if (!node->active) {
            continue;
        }
        
        /* Calculate base importance factors */
        float edge_factor = 0.0f;
        int incoming_edges = 0, outgoing_edges = 0;
        
        for (int j = 0; j < model->num_edges; j++) {
            if (model->edges[j].dst == i) {
                incoming_edges++;
                edge_factor += model->edges[j].weight * model->edges[j].attention_score;
            }
            if (model->edges[j].src == i) {
                outgoing_edges++;
                edge_factor += model->edges[j].weight * model->edges[j].attention_score * 0.5f;
            }
        }
        
        /* Calculate recency factor - FIXED: Use correct field name */
        float access_factor = (node->last_accessed_iteration >= model->current_iteration - 5) ? 1.2f : 0.8f;
        
        /* Calculate connectivity factor */
        float connectivity_factor = 1.0f + 0.1f * (incoming_edges + outgoing_edges);
        
        /* Update importance score with weighted combination */
        float new_importance = 0.4f * node->importance_score +                /* Current importance (momentum) */
                              0.3f * (edge_factor / fmaxf(1.0f, incoming_edges + outgoing_edges)) + /* Edge contribution */
                              0.2f * access_factor +                          /* Recency factor */
                              0.1f * connectivity_factor;                     /* Connectivity bonus */
        
        /* Ensure importance stays in valid range */
        node->importance_score = fmaxf(0.0f, fminf(1.0f, new_importance));
        
        /* Update attention centrality based on connections */
        node->attention_centrality = 0.8f * node->attention_centrality + 
                                   0.2f * (float)(incoming_edges + outgoing_edges) / 
                                   fmaxf(1.0f, model->num_edges / (float)model->num_nodes);
    }
}

/* =============================================================================
 * MAIN PROCESSING COORDINATION WITH LAZY LOADING
 * =============================================================================
 */

/* Main sequence processing function - coordinates all modules with lazy loading */
void galileo_process_sequence(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    if (!model || !tokens || num_tokens <= 0) {
        return;
    }
    
    printf("\n🚀 === Enhanced Galileo v42 Processing %d tokens ===\n", num_tokens);
    
    /* Phase 1: Add all tokens to the graph (core functionality) */
    for (int i = 0; i < num_tokens; i++) {
        int node_idx = galileo_add_token(model, tokens[i]);
        if (node_idx >= 0) {
            printf("📝 Added token '%s' as node %d (importance: %.2f)\n", 
                   tokens[i], node_idx, model->nodes[node_idx].importance_score);
        }
    }
    printf("📝 All tokens added to graph.\n");
    
    /* Phase 2: Iterative processing with lazy-loaded modules */
    for (int iter = 0; iter < model->max_iterations; iter++) {
        model->current_iteration = iter;
        printf("\n🔄 Iteration %d/%d...\n", iter + 1, model->max_iterations);
        
        /* Try to use graph module for message passing */
        if (call_graph_function_lazy("galileo_message_passing_iteration", model)) {
            printf("📡 Running message passing (graph module)...\n");
        } else {
            printf("⏭️  Skipping message passing (graph module not available)\n");
        }
        
        /* Try to use symbolic module for reasoning */
        if (call_symbolic_function_lazy("galileo_enhanced_symbolic_inference_safe", model)) {
            printf("🧠 Running symbolic inference (symbolic module)...\n");
            model->total_symbolic_calls++;
        } else {
            printf("⏭️  Skipping symbolic reasoning (symbolic module not available)\n");
        }
        
        /* Try to use heuristic compiler for fact extraction */
        if (call_heuristic_function_lazy("extract_facts_with_heuristic_compiler", model, tokens, num_tokens)) {
            printf("🧬 Running heuristic fact extraction (heuristic module)...\n");
        } else {
            printf("⏭️  Skipping heuristic extraction (heuristic module not available)\n");
        }
        
        /* Try to use memory module for compression */
        if (call_memory_function_lazy("galileo_adaptive_compression", model)) {
            printf("🗜️  Running adaptive compression (memory module)...\n");
        } else {
            printf("⏭️  Skipping compression (memory module not available)\n");
        }
        
        /* Update importance scores (core functionality) */
        galileo_update_importance_scores(model);
        
        /* Compute statistics (core functionality) */
        if (iter % 3 == 0) {  /* Every 3rd iteration */
            galileo_compute_graph_stats(model);
        }
    }
    
    printf("\n✅ Processing complete after %d iterations!\n", model->max_iterations);
    printf("🎯 Final model state: %d nodes, %d edges, %d facts\n", 
           model->num_nodes, model->num_edges, model->num_facts);
}

/* =============================================================================
 * PERFORMANCE AND DEBUGGING
 * =============================================================================
 */

/* Get performance statistics */
ModulePerformanceStats galileo_get_module_performance_stats(void) {
    return g_performance_stats;
}

/* Reset performance counters */
void galileo_reset_module_performance_stats(void) {
    memset(&g_performance_stats, 0, sizeof(g_performance_stats));
}
