/* =============================================================================
 * galileo/src/core/galileo_core.h - Core Module Public API
 * 
 * UPDATED for enhanced lazy loading and fixed function signatures.
 * Public header for the core Galileo module containing fundamental types,
 * structures, and function declarations for model lifecycle management.
 * 
 * This header defines ALL the core functions actually implemented in the
 * core module, with correct signatures and complete API coverage.
 * =============================================================================
 */

#ifndef GALILEO_CORE_H
#define GALILEO_CORE_H

#include "galileo_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

/* =============================================================================
 * CORE FUNCTION DECLARATIONS
 * =============================================================================
 */

/* Model lifecycle management */
GalileoModel* galileo_init(void);
void galileo_destroy(GalileoModel* model);

/* Basic node and token operations */
int galileo_add_token(GalileoModel* model, const char* token_text);

/* Enhanced token embedding - FIXED SIGNATURE to match implementation */
float* get_enhanced_token_embedding(GalileoModel* model, const char* token_text, 
                                   int context_position);

/* Model analysis and statistics */
void galileo_compute_graph_stats(GalileoModel* model);
void galileo_update_importance_scores(GalileoModel* model);

/* Processing coordination - UPDATED for lazy loading */
void galileo_process_sequence(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens);

/* Enhanced hash function for token processing */
uint32_t enhanced_hash(const char* str);

/* =============================================================================
 * UTILITY FUNCTIONS - ADDED ALL MISSING DECLARATIONS
 * =============================================================================
 */

/* Safe string operations */
void safe_strcpy(char* dest, const char* src, size_t dest_size);

/* Random number generation */
float random_float(void);
float random_float_range(void);

/* Vector operations with NaN protection */
float vector_dot_product(const float* a, const float* b, int dim);
float vector_magnitude(const float* vec, int dim);
/* REMOVED: cosine_similarity - should be in utils module */
void vector_normalize(float* vec, int dim);
void vector_add_scaled(float* dest, const float* src, float scale, int dim);
void vector_copy(float* dest, const float* src, int dim);
void vector_zero(float* vec, int dim);

/* =============================================================================
 * DYNAMIC MODULE FUNCTION CALLING - COMPLETE API
 * =============================================================================
 */

/* External declaration for lazy loading function from main module */
extern int ensure_module_loaded(const char* module_name);

/* Module availability checking */
extern int galileo_is_module_loaded(const char* module_name);

/* Lower-level module function calling */
void* get_module_function(const char* module_name, const char* function_name);

/* Safe wrappers that handle lazy loading automatically */
int call_graph_function_lazy(const char* function_name, GalileoModel* model);
int call_symbolic_function_lazy(const char* function_name, GalileoModel* model);
int call_memory_function_lazy(const char* function_name, GalileoModel* model);
int call_heuristic_function_lazy(const char* function_name, GalileoModel* model, 
                                char tokens[][MAX_TOKEN_LEN], int num_tokens);
int call_utils_function_lazy(const char* function_name, GalileoModel* model);

/* =============================================================================
 * MODULE INTEGRATION STATUS CHECKING
 * =============================================================================
 */

/* Check if specific modules are available for use */
int galileo_has_graph_module(void);
int galileo_has_symbolic_module(void);
int galileo_has_memory_module(void);
int galileo_has_heuristic_module(void);
int galileo_has_utils_module(void);

/* =============================================================================
 * ENHANCED PROCESSING COORDINATION
 * =============================================================================
 */

/* Processing phases with automatic module loading */
typedef enum {
    GALILEO_PHASE_TOKENIZATION,
    GALILEO_PHASE_GRAPH_CONSTRUCTION,
    GALILEO_PHASE_MESSAGE_PASSING,
    GALILEO_PHASE_SYMBOLIC_REASONING,
    GALILEO_PHASE_HEURISTIC_EXTRACTION,
    GALILEO_PHASE_MEMORY_COMPRESSION,
    GALILEO_PHASE_STATISTICS
} GalileoProcessingPhase;

/* Execute a specific processing phase with lazy loading */
int galileo_execute_phase(GalileoModel* model, GalileoProcessingPhase phase, 
                         char tokens[][MAX_TOKEN_LEN], int num_tokens);

/* =============================================================================
 * PERFORMANCE AND DEBUGGING
 * =============================================================================
 */

/* Performance monitoring for module loading */
typedef struct {
    int module_load_count;
    int module_function_calls;
    double total_module_load_time_ms;
    double total_function_call_time_ms;
    char last_loaded_module[64];
    char most_used_module[64];
} ModulePerformanceStats;

/* Get performance statistics */
ModulePerformanceStats galileo_get_module_performance_stats(void);

/* Reset performance counters */
void galileo_reset_module_performance_stats(void);

/* =============================================================================
 * MODULE INTERFACE FOR DYNAMIC LOADING
 * =============================================================================
 */

/* Module info structure for dynamic loading */
typedef struct {
    const char* name;
    const char* version;
    int (*init_func)(void);
    void (*cleanup_func)(void);
} CoreModuleInfo;

/* Module info instance - exported for module loader */
extern CoreModuleInfo core_module_info;

#endif /* GALILEO_CORE_H */
