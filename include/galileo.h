/* =============================================================================
 * galileo/include/galileo.h - Master Public Header
 * 
 * Main public header for the Galileo v42 Graph-and-Logic Integrated Language
 * Engine. This header provides a unified interface to all Galileo modules
 * and is the single include file that external projects need.
 * 
 * This header is installed to the system include path and provides access
 * to the complete Galileo API through a clean, modular interface.
 * =============================================================================
 */

#ifndef GALILEO_H
#define GALILEO_H

/* =============================================================================
 * SYSTEM INCLUDES
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>

/* Dynamic loading support (for module system) */
#ifdef __unix__
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

/* Command line parsing support */
#ifdef __GLIBC__
#include <getopt.h>
#endif

/* =============================================================================
 * GALILEO VERSION INFORMATION
 * =============================================================================
 */

#define GALILEO_VERSION_MAJOR 42
#define GALILEO_VERSION_MINOR 1
#define GALILEO_VERSION_PATCH 0
#define GALILEO_VERSION_STRING "42.1.0"
#define GALILEO_API_VERSION "42.1"

/* Version checking macros */
#define GALILEO_VERSION_CHECK(major, minor, patch) \
    ((GALILEO_VERSION_MAJOR > (major)) || \
     (GALILEO_VERSION_MAJOR == (major) && GALILEO_VERSION_MINOR > (minor)) || \
     (GALILEO_VERSION_MAJOR == (major) && GALILEO_VERSION_MINOR == (minor) && GALILEO_VERSION_PATCH >= (patch)))

/* =============================================================================
 * GALILEO CORE MODULE HEADERS
 * 
 * Core type definitions and fundamental structures that all other modules
 * depend on. These must be included first.
 * =============================================================================
 */

/* Fundamental type definitions - MUST be first */
#include "../src/core/galileo_types.h"

/* Core model lifecycle and basic operations */
#include "../src/core/galileo_core.h"

/* =============================================================================
 * GALILEO FUNCTIONAL MODULE HEADERS
 * 
 * The main functional modules that implement Galileo's capabilities.
 * Order matters due to dependencies.
 * =============================================================================
 */

/* Graph neural network operations */
#include "../src/graph/galileo_graph.h"

/* Symbolic reasoning and logical inference */
#include "../src/symbolic/galileo_symbolic.h"

/* Memory management and contextual addressing */
#include "../src/memory/galileo_memory.h"

/* Utilities, I/O, and helper functions */
#include "../src/utils/galileo_utils.h"

/* =============================================================================
 * GALILEO APPLICATION INTERFACE
 * 
 * CLI interface and main application entry points.
 * =============================================================================
 */

/* Command-line interface and main application */
#include "../src/main/galileo_main.h"

/* =============================================================================
 * GALILEO API CONVENIENCE MACROS
 * =============================================================================
 */

/* Quick model creation and destruction */
#define GALILEO_CREATE_MODEL() galileo_init()
#define GALILEO_DESTROY_MODEL(model) do { \
    if (model) { \
        galileo_destroy(model); \
        model = NULL; \
    } \
} while(0)

/* Safe model validation */
#define GALILEO_VALIDATE_MODEL(model) \
    ((model) && galileo_validate_model(model))

/* Quick processing macros */
#define GALILEO_PROCESS_TEXT(model, text) do { \
    if (GALILEO_VALIDATE_MODEL(model) && (text)) { \
        int token_count; \
        char** tokens = tokenize_input(text, &token_count); \
        if (tokens && token_count > 0) { \
            char token_array[token_count][MAX_TOKEN_LEN]; \
            for (int i = 0; i < token_count; i++) { \
                strncpy(token_array[i], tokens[i], MAX_TOKEN_LEN - 1); \
                token_array[i][MAX_TOKEN_LEN - 1] = '\0'; \
            } \
            galileo_process_sequence(model, token_array, token_count); \
            free_tokens(tokens, token_count); \
        } \
    } \
} while(0)

/* Module availability checking */
#define GALILEO_HAS_GRAPH_MODULE() is_module_loaded("graph")
#define GALILEO_HAS_SYMBOLIC_MODULE() is_module_loaded("symbolic")
#define GALILEO_HAS_MEMORY_MODULE() is_module_loaded("memory")
#define GALILEO_HAS_UTILS_MODULE() is_module_loaded("utils")

/* =============================================================================
 * GALILEO CONFIGURATION AND BUILD INFO
 * =============================================================================
 */

/* Build configuration detection */
#ifdef NDEBUG
#define GALILEO_BUILD_TYPE "Release"
#else
#define GALILEO_BUILD_TYPE "Debug"
#endif

#ifdef GALILEO_EXPOSE_INTERNAL_REASONING
#define GALILEO_SYMBOLIC_INTERNALS_AVAILABLE 1
#else
#define GALILEO_SYMBOLIC_INTERNALS_AVAILABLE 0
#endif

#ifdef GALILEO_EXPOSE_INTERNAL_MEMORY
#define GALILEO_MEMORY_INTERNALS_AVAILABLE 1
#else
#define GALILEO_MEMORY_INTERNALS_AVAILABLE 0
#endif

#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
#define GALILEO_ADVANCED_UTILS_AVAILABLE 1
#else
#define GALILEO_ADVANCED_UTILS_AVAILABLE 0
#endif

/* Compile-time feature detection */
#if defined(__GNUC__) || defined(__clang__)
#define GALILEO_COMPILER_SUPPORTS_BUILTIN_FUNCTIONS 1
#else
#define GALILEO_COMPILER_SUPPORTS_BUILTIN_FUNCTIONS 0
#endif

/* =============================================================================
 * GALILEO API INFORMATION FUNCTIONS
 * =============================================================================
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Get version information */
static inline const char* galileo_get_version_string(void) {
    return GALILEO_VERSION_STRING;
}

static inline int galileo_get_version_major(void) {
    return GALILEO_VERSION_MAJOR;
}

static inline int galileo_get_version_minor(void) {
    return GALILEO_VERSION_MINOR;
}

static inline int galileo_get_version_patch(void) {
    return GALILEO_VERSION_PATCH;
}

/* Get build information */
static inline const char* galileo_get_build_type(void) {
    return GALILEO_BUILD_TYPE;
}

/* Get API capabilities */
static inline int galileo_has_symbolic_internals(void) {
    return GALILEO_SYMBOLIC_INTERNALS_AVAILABLE;
}

static inline int galileo_has_memory_internals(void) {
    return GALILEO_MEMORY_INTERNALS_AVAILABLE;
}

static inline int galileo_has_advanced_utils(void) {
    return GALILEO_ADVANCED_UTILS_AVAILABLE;
}

/* Print complete Galileo information */
static inline void galileo_print_info(FILE* output) {
    if (!output) output = stdout;
    
    fprintf(output, "Galileo Graph-and-Logic Integrated Language Engine\n");
    fprintf(output, "Version: %s (%s build)\n", GALILEO_VERSION_STRING, GALILEO_BUILD_TYPE);
    fprintf(output, "API Version: %s\n", GALILEO_API_VERSION);
    fprintf(output, "\nAvailable modules:\n");
    fprintf(output, "  Core: Always available\n");
    fprintf(output, "  Graph: %s\n", GALILEO_HAS_GRAPH_MODULE() ? "Loaded" : "Not loaded");
    fprintf(output, "  Symbolic: %s%s\n", 
            GALILEO_HAS_SYMBOLIC_MODULE() ? "Loaded" : "Not loaded",
            GALILEO_SYMBOLIC_INTERNALS_AVAILABLE ? " (with internals)" : "");
    fprintf(output, "  Memory: %s%s\n", 
            GALILEO_HAS_MEMORY_MODULE() ? "Loaded" : "Not loaded",
            GALILEO_MEMORY_INTERNALS_AVAILABLE ? " (with internals)" : "");
    fprintf(output, "  Utils: %s%s\n", 
            GALILEO_HAS_UTILS_MODULE() ? "Loaded" : "Not loaded",
            GALILEO_ADVANCED_UTILS_AVAILABLE ? " (with advanced features)" : "");
    
    fprintf(output, "\nConfiguration limits:\n");
    fprintf(output, "  Max tokens: %d\n", MAX_TOKENS);
    fprintf(output, "  Max edges: %d\n", MAX_EDGES);
    fprintf(output, "  Max facts: %d\n", MAX_FACTS);
    fprintf(output, "  Max memory slots: %d\n", MAX_MEMORY_SLOTS);
    fprintf(output, "  Embedding dimension: %d\n", EMBEDDING_DIM);
}

#ifdef __cplusplus
}
#endif

/* =============================================================================
 * GALILEO QUICK START EXAMPLE
 * =============================================================================
 */

/*
 * Quick Start Example:
 * 
 * #include <galileo.h>
 * 
 * int main() {
 *     // Create model
 *     GalileoModel* model = GALILEO_CREATE_MODEL();
 *     if (!model) return 1;
 *     
 *     // Process some text
 *     GALILEO_PROCESS_TEXT(model, "All birds can fly. Penguins are birds.");
 *     
 *     // Add symbolic facts
 *     galileo_add_fact(model, "penguin", "cannot", "fly", 0.95f);
 *     
 *     // Run reasoning
 *     galileo_enhanced_symbolic_inference_safe(model);
 *     
 *     // Show results
 *     print_facts(model, stdout);
 *     
 *     // Cleanup
 *     GALILEO_DESTROY_MODEL(model);
 *     return 0;
 * }
 */

#endif /* GALILEO_H */
