/* =============================================================================
 * galileo/src/core/galileo_core.c - Core Module Implementation
 * 
 * Hot-loadable shared library implementing the core Galileo model structure,
 * lifecycle management, and basic operations. This is the foundation that
 * all other modules build upon.
 * 
 * Integrated with the dynamic module loading system - this module can be
 * loaded on-demand and provides the essential functions needed to create
 * and manage Galileo models.
 * 
 * 🎯 FIXED: Added dynamic module function calling system to replace TODOs!
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
#include <dlfcn.h>  /* 🎯 NEW: Added for dynamic function loading */
#include <unistd.h> /* 🎯 NEW: Added for getcwd() */

/* Thread support - portable across UNIX systems */
#if defined(_POSIX_THREADS) || defined(__APPLE__) || defined(__linux__)
    #include <pthread.h>
    #define GALILEO_HAS_PTHREAD 1
#else
    #define GALILEO_HAS_PTHREAD 0
#endif

/* Thread-local storage - portable implementation */
#if defined(__GNUC__) && GALILEO_HAS_PTHREAD
    #define GALILEO_THREAD_LOCAL __thread
#elif defined(_MSC_VER)
    #define GALILEO_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define GALILEO_THREAD_LOCAL _Thread_local
#else
    #define GALILEO_THREAD_LOCAL static
    #warning "Thread-local storage not supported, using static (not thread-safe)"
#endif

/* =============================================================================
 * MODULE METADATA AND INITIALIZATION
 * =============================================================================
 */

static int core_module_initialized = 0;
static int active_model_count = 0;

/* Module initialization */
static int core_module_init(void) {
    if (core_module_initialized) {
        return 0;  /* Already initialized */
    }
    
    fprintf(stderr, "🧠 Core module v42.1 initializing...\n");
    
    /* Initialize any core subsystems */
    active_model_count = 0;
    
    /* Seed random number generator for embedding initialization */
    srand((unsigned int)time(NULL));
    
    core_module_initialized = 1;
    fprintf(stderr, "✅ Core module ready! Multi-model support enabled.\n");
    return 0;
}

/* Module cleanup */
static void core_module_cleanup(void) {
    if (!core_module_initialized) {
        return;
    }
    
    fprintf(stderr, "🧠 Core module shutting down...\n");
    
    if (active_model_count > 0) {
        fprintf(stderr, "⚠️  Warning: %d models still active during shutdown\n", active_model_count);
    }
    
    core_module_initialized = 0;
}

/* Module info structure for dynamic loading */
CoreModuleInfo core_module_info = {
    .name = "core",
    .version = "42.1", 
    .init_func = core_module_init,
    .cleanup_func = core_module_cleanup
};

/* =============================================================================
 * DYNAMIC MODULE FUNCTION CALLING HELPERS
 * 🎯 NEW: Helper functions to safely call functions from loaded modules
 * =============================================================================
 */

/* Get a function pointer from a loaded module */
static void* get_module_function(const char* module_name, const char* function_name) {
    /* Check if module is loaded */
    if (!galileo_is_module_loaded(module_name)) {
        return NULL;
    }
    
    /* Get the module handle directly using dlopen with RTLD_NOLOAD */
    /* This gets the handle without loading, since module is already loaded */
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
            fprintf(stderr, "⚠️  Could not get handle for module '%s': %s\n", module_name, dlerror());
            return NULL;
        }
    }
    
    /* Get the function pointer */
    void* func_ptr = dlsym(module_handle, function_name);
    if (!func_ptr) {
        fprintf(stderr, "⚠️  Function '%s' not found in module '%s': %s\n", 
                function_name, module_name, dlerror());
        dlclose(module_handle);
        return NULL;
    }
    
    /* Don't close the handle since the module should stay loaded */
    return func_ptr;
}

/* Safe wrapper to call graph module functions */
static int call_graph_function(const char* function_name, GalileoModel* model) {
    typedef void (*graph_func_t)(GalileoModel*);
    graph_func_t func = (graph_func_t)get_module_function("graph", function_name);
    
    if (func) {
        func(model);
        return 1;  /* Success */
    }
    
    return 0;  /* Function not available */
}

/* Safe wrapper to call symbolic module functions */
static int call_symbolic_function(const char* function_name, GalileoModel* model) {
    typedef void (*symbolic_func_t)(GalileoModel*);
    symbolic_func_t func = (symbolic_func_t)get_module_function("symbolic", function_name);
    
    if (func) {
        func(model);
        return 1;  /* Success */
    }
    
    return 0;  /* Function not available */
}

/* Safe wrapper to call memory module functions */
static int call_memory_function(const char* function_name, GalileoModel* model) {
    typedef void (*memory_func_t)(GalileoModel*);
    memory_func_t func = (memory_func_t)get_module_function("memory", function_name);
    
    if (func) {
        func(model);
        return 1;  /* Success */
    }
    
    return 0;  /* Function not available */
}

/* Safe wrapper to call heuristic module functions */
static int call_heuristic_function(const char* function_name, GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    typedef int (*heuristic_func_t)(GalileoModel*, char[][MAX_TOKEN_LEN], int);
    heuristic_func_t func = (heuristic_func_t)get_module_function("heuristic", function_name);
    
    if (func) {
        return func(model, tokens, num_tokens);
    }
    
    return 0;  /* Function not available */
}

/* =============================================================================
 * UTILITY FUNCTIONS
 * =============================================================================
 */

/* Enhanced hash function for token processing */
uint32_t enhanced_hash(const char* str) {
    if (!str) return 0;
    
    uint32_t hash = 5381;
    for (const char* c = str; *c; c++) {
        hash = ((hash << 5) + hash) + (unsigned char)*c;
    }
    return hash;
}

/* Generate enhanced token embedding with positional encoding - THREAD-SAFE */
float* get_enhanced_token_embedding(GalileoModel* model, const char* token_text, int context_position, float* output_buffer) {
    if (!model || !token_text || context_position >= MAX_TOKENS || !output_buffer) {
        return NULL;
    }
    
    /* Find or create the token node */
    int token_idx = -1;
    for (int i = 1; i < model->num_nodes; i++) {  /* Skip global node at index 0 */
        if (!model->nodes[i].is_summary && !model->nodes[i].is_global &&
            strcmp(model->nodes[i].token_text, token_text) == 0) {
            token_idx = i;
            break;
        }
    }
    
    if (token_idx == -1) {
        /* Token not found, return NULL to trigger creation elsewhere */
        return NULL;
    }
    
    /* Apply positional encoding to the base embedding - using caller's buffer */
    memcpy(output_buffer, model->nodes[token_idx].identity_embedding, EMBEDDING_DIM * sizeof(float));
    
    /* Add positional encoding */
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        if (i % 2 == 0) {
            output_buffer[i] += 0.1f * sinf(context_position / powf(10000.0f, 2.0f * i / EMBEDDING_DIM));
        } else {
            output_buffer[i] += 0.1f * cosf(context_position / powf(10000.0f, 2.0f * i / EMBEDDING_DIM));
        }
    }
    
    return output_buffer;
}

/* Initialize random embedding for a node - PORTABLE THREAD-SAFE */
static void init_random_embedding(float* embedding, int dim) {
    /* Portable thread-safe random number generation */
#if GALILEO_HAS_PTHREAD && (defined(__linux__) || defined(__GLIBC__))
    /* Linux/glibc: Use rand_r() with thread-local seed */
    static GALILEO_THREAD_LOCAL unsigned int seed = 0;
    
    if (seed == 0) {
        seed = (unsigned int)time(NULL) ^ (unsigned int)pthread_self();
    }
    
    for (int i = 0; i < dim; i++) {
        embedding[i] = ((float)rand_r(&seed) / RAND_MAX - 0.5f) * 2.0f;
    }
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    /* BSD systems: Use arc4random() (thread-safe by design) */
    for (int i = 0; i < dim; i++) {
        embedding[i] = ((float)arc4random() / (float)0xFFFFFFFF - 0.5f) * 2.0f;
    }
#elif GALILEO_HAS_PTHREAD
    /* Other POSIX systems: Use mutex-protected rand() */
    static pthread_mutex_t rand_mutex = PTHREAD_MUTEX_INITIALIZER;
    static int seeded = 0;
    
    pthread_mutex_lock(&rand_mutex);
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < dim; i++) {
        embedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    pthread_mutex_unlock(&rand_mutex);
#else
    /* Fallback: Basic rand() (not thread-safe) */
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }
    
    for (int i = 0; i < dim; i++) {
        embedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
#endif
    
    /* Normalize the embedding */
    float norm = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);
    
    if (norm > 0.0f) {
        for (int i = 0; i < dim; i++) {
            embedding[i] /= norm;
        }
    }
}

/* =============================================================================
 * MODEL LIFECYCLE MANAGEMENT
 * =============================================================================
 */

/* Initialize a new Galileo model */
GalileoModel* galileo_init(void) {
    printf("🚀 Initializing Galileo v42 model...\n");
    
    GalileoModel* model = calloc(1, sizeof(GalileoModel));
    if (!model) {
        fprintf(stderr, "❌ Failed to allocate memory for Galileo model\n");
        return NULL;
    }
    
    /* Enhanced default parameters */
    model->similarity_threshold = 0.85f;
    model->attention_threshold = 0.75f;
    model->compression_threshold = 0.9f;
    model->importance_decay = 0.95f;
    model->max_iterations = 8;
    model->max_edges_per_iteration = 50;
    
    /* Initialize global context node (always at index 0) */
    model->nodes[0].is_global = 1;
    model->nodes[0].is_summary = 0;
    strncpy(model->nodes[0].token_text, "GLOBAL_CONTEXT", MAX_TOKEN_LEN - 1);
    model->nodes[0].token_text[MAX_TOKEN_LEN - 1] = '\0';
    init_random_embedding(model->nodes[0].identity_embedding, EMBEDDING_DIM);
    init_random_embedding(model->nodes[0].context_embedding, EMBEDDING_DIM);
    init_random_embedding(model->nodes[0].temporal_embedding, EMBEDDING_DIM);
    model->nodes[0].importance_score = 1.0f;
    model->nodes[0].active = 1;
    model->nodes[0].compression_level = 0;
    model->nodes[0].last_accessed_iteration = 0;
    model->nodes[0].node_id = 0;
    model->global_node_idx = 0;
    model->num_nodes = 1;
    
    /* Initialize edge hash table */
    for (int i = 0; i < EDGE_HASH_SIZE; i++) {
        model->edge_hash[i] = NULL;
    }
    
    /* Initialize dynamic conflict array */
    model->resolved_conflicts = NULL;
    model->num_resolved_conflicts = 0;
    model->resolved_conflicts_capacity = 0;
    
    /* Update active model counter */
    active_model_count++;
    
    printf("🌐 Created global context node (id: 0)\n");
    printf("✅ Galileo model initialized successfully! (Model #%d)\n", active_model_count);
    printf("   Parameters: sim_thresh=%.2f, att_thresh=%.2f, max_iter=%d\n",
           model->similarity_threshold, model->attention_threshold, model->max_iterations);
    
    return model;
}

/* Destroy a Galileo model and clean up all resources */
void galileo_destroy(GalileoModel* model) {
    if (!model) return;
    
    printf("🔥 Destroying Galileo model...\n");
    
    /* Clean up edge hash table */
    for (int i = 0; i < EDGE_HASH_SIZE; i++) {
        EdgeHashEntry* entry = model->edge_hash[i];
        while (entry) {
            EdgeHashEntry* next = entry->next;
            free(entry);
            entry = next;
        }
        model->edge_hash[i] = NULL;
    }
    
    /* Clean up dynamic conflict descriptions */
    if (model->resolved_conflicts) {
        for (int i = 0; i < model->num_resolved_conflicts; i++) {
            if (model->resolved_conflicts[i]) {
                if (model->resolved_conflicts[i]->data) {
                    free(model->resolved_conflicts[i]->data);
                }
                free(model->resolved_conflicts[i]);
            }
        }
        free(model->resolved_conflicts);
        model->resolved_conflicts = NULL;
    }
    
    /* Print final statistics */
    printf("📊 Final stats: %d nodes, %d edges, %d facts, %d compressions\n",
           model->num_nodes, model->num_edges, model->num_facts, model->total_compressions);
    
    /* Zero out the structure for security */
    memset(model, 0, sizeof(GalileoModel));
    
    /* Free the model */
    free(model);
    
    /* Update global counter */
    active_model_count--;
    
    printf("✅ Model destroyed. (%d models remaining)\n", active_model_count);
}

/* =============================================================================
 * TOKEN AND NODE MANAGEMENT
 * =============================================================================
 */

/* Add a token to the model and create corresponding node */
int galileo_add_token(GalileoModel* model, const char* token_text) {
    if (!model || !token_text || model->num_nodes >= MAX_TOKENS) {
        return -1;
    }
    
    /* Check if token already exists */
    for (int i = 1; i < model->num_nodes; i++) {  /* Skip global node */
        if (!model->nodes[i].is_summary && !model->nodes[i].is_global &&
            strcmp(model->nodes[i].token_text, token_text) == 0) {
            return i;  /* Token already exists, return its index */
        }
    }
    
    /* Create new token node */
    int node_idx = model->num_nodes;
    GraphNode* node = &model->nodes[node_idx];
    
    /* Initialize node properties */
    node->is_global = 0;
    node->is_summary = 0;
    strncpy(node->token_text, token_text, MAX_TOKEN_LEN - 1);
    node->token_text[MAX_TOKEN_LEN - 1] = '\0';
    node->node_id = node_idx;
    
    /* Initialize embeddings */
    init_random_embedding(node->identity_embedding, EMBEDDING_DIM);
    init_random_embedding(node->context_embedding, EMBEDDING_DIM);
    init_random_embedding(node->temporal_embedding, EMBEDDING_DIM);
    
    /* Set initial properties */
    node->importance_score = 0.5f;  /* Default importance */
    node->active = 1;
    node->compression_level = 0;
    node->last_accessed_iteration = model->current_iteration;
    
    /* Add to vocabulary if there's space */
    if (model->vocab_size < MAX_VOCAB_SIZE) {
        VocabEntry* entry = &model->vocabulary[model->vocab_size];
        strncpy(entry->token, token_text, MAX_TOKEN_LEN - 1);
        entry->token[MAX_TOKEN_LEN - 1] = '\0';
        entry->frequency = 1;
        entry->node_id = node_idx;
        entry->importance = node->importance_score;
        model->vocab_size++;
    }
    
    model->num_nodes++;
    
    return node_idx;
}

/* =============================================================================
 * MODEL ANALYSIS AND STATISTICS
 * =============================================================================
 */

/* Compute comprehensive graph statistics */
void galileo_compute_graph_stats(GalileoModel* model) {
    if (!model) return;
    
    printf("\n📊 === Graph Statistics ===\n");
    
    /* Basic counts */
    printf("Nodes: %d, Edges: %d, Facts: %d\n", 
           model->num_nodes, model->num_edges, model->num_facts);
    
    /* Compute average degree */
    if (model->num_nodes > 0) {
        model->avg_node_degree = (float)(2 * model->num_edges) / model->num_nodes;
        printf("Average degree: %.2f\n", model->avg_node_degree);
    }
    
    /* Vocabulary statistics */
    printf("Vocabulary size: %d/%d\n", model->vocab_size, MAX_VOCAB_SIZE);
    
    /* Memory usage */
    printf("Memory slots used: %d/%d\n", model->num_memory_slots, MAX_MEMORY_SLOTS);
    
    /* Performance metrics */
    printf("Total edges added: %d\n", model->total_edges_added);
    printf("Total compressions: %d\n", model->total_compressions);
    printf("Total symbolic calls: %d\n", model->total_symbolic_calls);
    
    /* Node type distribution */
    int token_nodes = 0, summary_nodes = 0, global_nodes = 0;
    for (int i = 0; i < model->num_nodes; i++) {
        if (model->nodes[i].is_global) {
            global_nodes++;
        } else if (model->nodes[i].is_summary) {
            summary_nodes++;
        } else {
            token_nodes++;
        }
    }
    printf("Node types: %d tokens, %d summaries, %d globals\n", 
           token_nodes, summary_nodes, global_nodes);
    
    /* Edge type distribution */
    int similarity_edges = 0, attention_edges = 0, semantic_edges = 0;
    for (int i = 0; i < model->num_edges; i++) {
        switch (model->edges[i].type) {
            case EDGE_SIMILARITY: similarity_edges++; break;
            case EDGE_ATTENTION: attention_edges++; break;
            case EDGE_SEMANTIC: semantic_edges++; break;
            default: break;
        }
    }
    printf("Edge types: %d similarity, %d attention, %d semantic\n",
           similarity_edges, attention_edges, semantic_edges);
}

/* Update importance scores for all nodes */
void galileo_update_importance_scores(GalileoModel* model) {
    if (!model) return;
    
    for (int i = 0; i < model->num_nodes; i++) {
        GraphNode* node = &model->nodes[i];
        
        /* Apply iteration-based decay */
        int age = model->current_iteration - node->last_accessed_iteration;
        float time_factor = expf(-age / 10.0f);  /* Decay over iterations */
        
        /* Apply access-based boost - use a simple access pattern */
        float access_factor = node->active ? 1.2f : 0.8f;
        
        /* Combine factors */
        node->importance_score = (node->importance_score * model->importance_decay) + 
                                (0.1f * time_factor * access_factor);
        
        /* Clamp to reasonable range */
        if (node->importance_score > 2.0f) node->importance_score = 2.0f;
        if (node->importance_score < 0.01f) node->importance_score = 0.01f;
    }
}

/* =============================================================================
 * PROCESSING COORDINATION
 * =============================================================================
 */

/* Main sequence processing function - coordinates all modules */
void galileo_process_sequence(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    if (!model || !tokens || num_tokens <= 0) {
        return;
    }
    
    printf("\n🚀 === Enhanced Galileo v42 Processing %d tokens ===\n", num_tokens);
    
    /* Phase 1: Add all tokens to the graph */
    for (int i = 0; i < num_tokens; i++) {
        int node_idx = galileo_add_token(model, tokens[i]);
        if (node_idx >= 0) {
            printf("📝 Added token '%s' as node %d (importance: %.2f)\n", 
                   tokens[i], node_idx, model->nodes[node_idx].importance_score);
        }
    }
    printf("📝 All tokens added to graph.\n");
    
    /* Phase 2: Iterative processing with multiple modules */
    for (int iter = 0; iter < model->max_iterations; iter++) {
        model->current_iteration = iter;
        printf("\n🔄 Iteration %d/%d...\n", iter + 1, model->max_iterations);
        
        /* 🎯 FIXED #1: Try to load and use graph module for message passing */
        if (galileo_is_module_loaded("graph")) {
            printf("📡 Running message passing (graph module)...\n");
            if (!call_graph_function("galileo_message_passing_iteration", model)) {
                printf("⚠️  Warning: Failed to call graph message passing function\n");
            }
        } else {
            printf("⏭️  Skipping message passing (graph module not loaded)\n");
        }
        
        /* 🎯 FIXED #2: Try to use symbolic module for reasoning */
        if (galileo_is_module_loaded("symbolic")) {
            printf("🧠 Running symbolic inference (symbolic module)...\n");
            if (call_symbolic_function("galileo_enhanced_symbolic_inference_safe", model)) {
                model->total_symbolic_calls++;
            } else {
                printf("⚠️  Warning: Failed to call symbolic reasoning function\n");
            }
        } else {
            printf("⏭️  Skipping symbolic reasoning (symbolic module not loaded)\n");
        }
        
        /* 🧬 NEW: Try to use heuristic compiler for fact extraction */
        if (galileo_is_module_loaded("heuristic")) {
            printf("🧬 Running heuristic fact extraction (heuristic module)...\n");
            if (call_heuristic_function("extract_facts_with_heuristic_compiler", model, tokens, num_tokens)) {
                printf("✅ Heuristic fact extraction completed\n");
            } else {
                printf("⚠️  Warning: Failed to call heuristic fact extraction\n");
            }
        } else {
            printf("⏭️  Skipping heuristic fact extraction (heuristic module not loaded)\n");
        }
        
        /* 🎯 FIXED #3: Try to use memory module for compression */
        if (galileo_is_module_loaded("memory") && iter % 3 == 0 && iter > 0) {
            printf("💾 Running memory compression (memory module)...\n");
            if (call_memory_function("galileo_adaptive_compression", model)) {
                model->total_compressions++;
            } else {
                printf("⚠️  Warning: Failed to call memory compression function\n");
            }
        }
        
        /* Update statistics */
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
 * MODEL VALIDATION AND INFORMATION
 * =============================================================================
 */

/* Validate model integrity */
int galileo_validate_model(GalileoModel* model) {
    if (!model) return 0;
    
    /* Check basic structure */
    if (model->num_nodes < 1 || model->num_nodes > MAX_TOKENS) return 0;
    if (model->num_edges < 0 || model->num_edges > MAX_EDGES) return 0;
    if (model->num_facts < 0 || model->num_facts > MAX_FACTS) return 0;
    
    /* Check global node exists */
    if (model->global_node_idx != 0) return 0;
    if (!model->nodes[0].is_global) return 0;
    
    /* Check parameter ranges */
    if (model->similarity_threshold < 0.0f || model->similarity_threshold > 1.0f) return 0;
    if (model->attention_threshold < 0.0f || model->attention_threshold > 1.0f) return 0;
    
    return 1;  /* Model is valid */
}

/* Get model information string */
int galileo_get_model_info(GalileoModel* model, char* info_buffer, size_t buffer_size) {
    if (!model || !info_buffer || buffer_size == 0) {
        return -1;
    }
    
    int written = snprintf(info_buffer, buffer_size,
        "Galileo Model v42.1\n"
        "Nodes: %d/%d, Edges: %d/%d, Facts: %d/%d\n"
        "Parameters: sim=%.2f, att=%.2f, max_iter=%d\n"
        "Performance: avg_degree=%.2f, compressions=%d\n"
        "Status: %s",
        model->num_nodes, MAX_TOKENS,
        model->num_edges, MAX_EDGES,
        model->num_facts, MAX_FACTS,
        model->similarity_threshold, model->attention_threshold, model->max_iterations,
        model->avg_node_degree, model->total_compressions,
        galileo_validate_model(model) ? "Valid" : "Invalid"
    );
    
    return (written > 0 && written < (int)buffer_size) ? 0 : -1;
}

/* =============================================================================
 * TESTING AND MULTI-MODEL SAFETY
 * =============================================================================
 */

/* Test multi-model safety and isolation */
void test_multi_model_safety(void) {
    printf("\n🧪 === Multi-Model Safety Tests ===\n");
    
    /* Test 1: Multiple model creation */
    printf("\n--- Test 1: Multiple Model Creation ---\n");
    GalileoModel* model1 = galileo_init();
    GalileoModel* model2 = galileo_init();
    
    if (model1 && model2 && model1 != model2) {
        printf("✅ Multiple models created successfully\n");
        printf("   Model1: %p, Model2: %p\n", (void*)model1, (void*)model2);
    } else {
        printf("❌ Failed to create multiple models\n");
        return;
    }
    
    /* Test 2: Model isolation */
    printf("\n--- Test 2: Model State Isolation ---\n");
    galileo_add_token(model1, "test_token_1");
    galileo_add_token(model2, "test_token_2");
    
    if (model1->num_nodes != model2->num_nodes) {
        printf("❌ Models sharing node count - isolation failed!\n");
    } else if (model1->num_nodes == 2 && model2->num_nodes == 2) {
        /* Both should have global node + 1 token = 2 nodes */
        printf("✅ Model isolation working correctly\n");
        printf("   Model1 nodes: %d, Model2 nodes: %d\n", 
               model1->num_nodes, model2->num_nodes);
    }
    
    /* Test 3: Clean destruction */
    printf("\n--- Test 3: Clean Destruction ---\n");
    galileo_destroy(model1);
    galileo_destroy(model2);
    
    /* Test 4: Fresh model after destruction */
    printf("\n--- Test 4: Fresh Model Creation ---\n");
    GalileoModel* model3 = galileo_init();
    
    if (model3 && model3->num_nodes == 1 && model3->num_facts == 0) {
        printf("✅ Fresh model starts clean\n");
    } else {
        printf("❌ Model not properly reset\n");
    }
    
    galileo_destroy(model3);
    printf("✅ Multi-model safety tests completed\n");
}
