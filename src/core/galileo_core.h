/* =============================================================================
 * galileo/src/core/galileo_core.h - Core Module Public API
 * 
 * Public header for the core Galileo module containing fundamental types,
 * structures, and function declarations for model lifecycle management.
 * 
 * This header defines the core GalileoModel structure and all the basic
 * operations that other modules depend on. It's the foundation of the
 * entire modular system.
 * =============================================================================
 */

#ifndef GALILEO_CORE_H
#define GALILEO_CORE_H

#include "galileo_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* =============================================================================
 * CORE FUNCTION DECLARATIONS
 * =============================================================================
 */

/* Model lifecycle management */
GalileoModel* galileo_init(void);
void galileo_destroy(GalileoModel* model);

/* Basic node and token operations */
int galileo_add_token(GalileoModel* model, const char* token_text);

/* Model analysis and statistics */
void galileo_compute_graph_stats(GalileoModel* model);
void galileo_update_importance_scores(GalileoModel* model);

/* Processing coordination */
void galileo_process_sequence(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens);

/* Model validation and information */
int galileo_validate_model(GalileoModel* model);
int galileo_get_model_info(GalileoModel* model, char* info_buffer, size_t buffer_size);

/* Testing and safety */
void test_multi_model_safety(void);

/* Module info for dynamic loading */
typedef struct {
    const char* name;
    const char* version;
    int (*init_func)(void);
    void (*cleanup_func)(void);
} CoreModuleInfo;

extern CoreModuleInfo core_module_info;

#endif /* GALILEO_CORE_H */
