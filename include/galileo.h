/* =============================================================================
 * galileo/include/galileo.h - Master Public Header (UPDATED)
 * 
 * UPDATED for enhanced lazy loading system with proper header management.
 * Main public header for the Galileo v42 Graph-and-Logic Integrated Language
 * Engine. This header provides a unified interface to all Galileo modules
 * and is the single include file that external projects need.
 * 
 * Fixed header ordering and ensured compatibility with lazy loading system.
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
#include <time.h>
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
#define GALILEO_VERSION_STRING "42.1.0-enhanced"
#define GALILEO_API_VERSION "42.1"
#define GALILEO_VERSION "42.1.0-enhanced"

/* Version checking macros */
#define GALILEO_VERSION_CHECK(major, minor, patch) \
    ((GALILEO_VERSION_MAJOR > (major)) || \
     (GALILEO_VERSION_MAJOR == (major) && GALILEO_VERSION_MINOR > (minor)) || \
     (GALILEO_VERSION_MAJOR == (major) && GALILEO_VERSION_MINOR == (minor) && GALILEO_VERSION_PATCH >= (patch)))

/* =============================================================================
 * GALILEO CORE MODULE HEADERS
 * 
 * Core type definitions and fundamental structures that all other modules
 * depend on. These must be included first and in the correct order.
 * =============================================================================
 */

/* Fundamental type definitions - MUST be first */
#include "../src/core/galileo_types.h"

/* Core model lifecycle and basic operations */
#include "../src/core/galileo_core.h"

/* Dynamic module loading system */
#include "../src/core/galileo_module_loader.h"

/* =============================================================================
 * GALILEO FUNCTIONAL MODULE HEADERS
 * 
 * The main functional modules that implement Galileo's capabilities.
 * Order matters due to dependencies. 
 * 
 * NOTE: In the lazy loading system, these headers are included for 
 * API definitions, but the actual modules are loaded on-demand at runtime.
 * =============================================================================
 */

/* Graph neural network operations */
#include "../src/graph/galileo_graph.h"

/* Symbolic reasoning and logic */
#include "../src/symbolic/galileo_symbolic.h"

/* Memory management and compression */
#include "../src/memory/galileo_memory.h"

/* Utility functions and I/O */
#include "../src/utils/galileo_utils.h"

/* Heuristic compiler with GA-derived fact extraction */
#include "../src/heuristic/galileo_heuristic_compiler.h"

/* =============================================================================
 * GALILEO MAIN APPLICATION INTERFACE
 * =============================================================================
 */

/* CLI interface and main application */
#include "../src/main/galileo_main.h"

/* =============================================================================
 * GALILEO PUBLIC API CONVENIENCE MACROS
 * =============================================================================
 */

/* Lazy loading helper macros for external users */
#define GALILEO_ENSURE_MODULE(name) ensure_module_loaded(name)
#define GALILEO_IS_MODULE_LOADED(name) galileo_is_module_loaded(name)

/* Version and feature checking */
#define GALILEO_HAS_LAZY_LOADING 1
#define GALILEO_HAS_HOT_PLUGGING 1
#define GALILEO_HAS_DYNAMIC_MODULES 1
#define GALILEO_HAS_19_YEAR_CACHE 1

/* =============================================================================
 * GALILEO FEATURE FLAGS
 * =============================================================================
 */

/* Module capability flags */
#define GALILEO_CAP_GRAPH_NEURAL_NETWORKS  (1 << 0)
#define GALILEO_CAP_SYMBOLIC_REASONING     (1 << 1)
#define GALILEO_CAP_MEMORY_COMPRESSION     (1 << 2)
#define GALILEO_CAP_HEURISTIC_COMPILATION  (1 << 3)
#define GALILEO_CAP_DYNAMIC_LOADING        (1 << 4)
#define GALILEO_CAP_HOT_PLUGGING           (1 << 5)

/* =============================================================================
 * GALILEO ERROR HANDLING
 * =============================================================================
 */

/* Error codes used throughout the system */
typedef enum {
    GALILEO_SUCCESS = 0,
    GALILEO_ERROR_GENERAL = 1,
    GALILEO_ERROR_ARGUMENTS = 2,
    GALILEO_ERROR_MODULE_LOADING = 3,
    GALILEO_ERROR_IO = 4,
    GALILEO_ERROR_MEMORY = 5,
    GALILEO_ERROR_SIGNAL = 130
} GalileoErrorCode;

/* =============================================================================
 * GALILEO INITIALIZATION AND CLEANUP
 * =============================================================================
 */

/* System-wide initialization and cleanup functions */
static inline int galileo_system_init(void) {
    return galileo_module_loader_init();
}

static inline void galileo_system_cleanup(void) {
    galileo_module_loader_cleanup();
}

/* =============================================================================
 * GALILEO MODULE DISCOVERY AND MANAGEMENT
 * =============================================================================
 */

/* High-level module management functions for external users */
static inline int galileo_discover_all_modules(void) {
    return galileo_discover_modules();
}

/* REMOVED: galileo_load_required_modules - conflicts with module_loader.h declaration */

/* =============================================================================
 * GALILEO CONVENIENCE FUNCTIONS
 * =============================================================================
 */

/* Quick start function for simple usage */
static inline GalileoModel* galileo_quick_start(void) {
    if (galileo_system_init() != 0) {
        return NULL;
    }
    
    if (load_bootstrap_modules() != 0) {  /* FIXED: Use direct function name */
        galileo_system_cleanup();
        return NULL;
    }
    
    return galileo_init();
}

/* Quick cleanup function */
static inline void galileo_quick_cleanup(GalileoModel* model) {
    if (model) {
        galileo_destroy(model);
    }
    galileo_system_cleanup();
}

#endif /* GALILEO_H */
