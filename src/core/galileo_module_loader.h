/* =============================================================================
 * galileo/src/core/galileo_module_loader.h - Dynamic Module Loading Engine
 * 
 * Public header for the hot-loading module system that provides JIT-style
 * on-demand loading of Galileo modules. This header defines all the public
 * APIs for dynamic module management.
 * 
 * Features exposed through this API:
 * - Runtime module discovery and registration
 * - On-demand loading with lazy initialization
 * - Module dependency tracking and resolution
 * - Thread-safe module management operations
 * - Module information queries and status reporting
 * - Hot-reload capabilities for development
 * =============================================================================
 */

#ifndef GALILEO_MODULE_LOADER_H
#define GALILEO_MODULE_LOADER_H

#include <stddef.h>
#include <stdio.h>
#include <time.h>

/* =============================================================================
 * CONFIGURATION CONSTANTS
 * =============================================================================
 */

/* Module discovery and management limits */
#ifndef MODULE_HASH_SIZE
#define MODULE_HASH_SIZE 64                 /* Hash table size for fast module lookup */
#endif

#ifndef MAX_SEARCH_PATHS  
#define MAX_SEARCH_PATHS 16                 /* Maximum module search directories */
#endif

#ifndef MAX_MODULE_NAME_LENGTH
#define MAX_MODULE_NAME_LENGTH 64           /* Maximum length of module names */
#endif

#ifndef MAX_MODULE_PATH_LENGTH
#define MAX_MODULE_PATH_LENGTH 512          /* Maximum length of module file paths */
#endif

#ifndef MAX_MODULE_VERSION_LENGTH
#define MAX_MODULE_VERSION_LENGTH 32        /* Maximum length of version strings */
#endif

#ifndef MAX_MODULE_ERROR_LENGTH
#define MAX_MODULE_ERROR_LENGTH 256         /* Maximum length of error messages */
#endif

#ifndef MAX_MODULE_DEPENDENCIES
#define MAX_MODULE_DEPENDENCIES 8           /* Maximum dependencies per module */
#endif

/* =============================================================================
 * PUBLIC DATA STRUCTURES
 * =============================================================================
 */

/* Module loading status and information */
typedef struct {
    char name[MAX_MODULE_NAME_LENGTH];          /* Module name */
    char path[MAX_MODULE_PATH_LENGTH];          /* Full path to .so file */
    char version[MAX_MODULE_VERSION_LENGTH];    /* Module version string */
    char error_message[MAX_MODULE_ERROR_LENGTH]; /* Last error message if any */
    
    int loaded;                                 /* Is module currently loaded? */
    int required;                               /* Is module required for operation? */
    int load_failed;                            /* Did the last load attempt fail? */
    int reference_count;                        /* Number of active references */
    
    long last_used_time;                        /* When was module last accessed (ms) */
    int dependency_count;                       /* Number of dependencies */
    char dependencies[MAX_MODULE_DEPENDENCIES][MAX_MODULE_NAME_LENGTH]; /* Dependency list */
} ModuleLoadInfo;

/* Module capability flags */
typedef enum {
    MODULE_CAP_NONE = 0,                        /* No special capabilities */
    MODULE_CAP_CORE = 1 << 0,                   /* Core functionality (required) */
    MODULE_CAP_GRAPH = 1 << 1,                  /* Graph neural network operations */
    MODULE_CAP_SYMBOLIC = 1 << 2,               /* Symbolic reasoning and logic */
    MODULE_CAP_MEMORY = 1 << 3,                 /* Memory management and compression */
    MODULE_CAP_UTILS = 1 << 4,                  /* Utility functions and I/O */
    MODULE_CAP_EXPERIMENTAL = 1 << 5,           /* Experimental features */
    MODULE_CAP_EXTENSIONS = 1 << 6,             /* Third-party extensions */
    MODULE_CAP_HOT_RELOAD = 1 << 7              /* Supports hot-reloading */
} ModuleCapabilities;

/* Module loading statistics */
typedef struct {
    int total_discovered;                       /* Total modules discovered */
    int currently_loaded;                       /* Currently loaded modules */
    int load_attempts;                          /* Total load attempts */
    int load_failures;                          /* Total load failures */
    int search_paths_configured;               /* Number of search paths */
    long total_memory_used;                     /* Total memory used by loaded modules */
    double average_load_time_ms;                /* Average module load time */
} ModuleLoadingStats;

/* =============================================================================
 * CORE MODULE LOADING API
 * =============================================================================
 */

/* System initialization and cleanup */
int galileo_module_loader_init(void);
void galileo_module_loader_cleanup(void);

/* Module search path management */
int galileo_add_module_search_path(const char* path);
int galileo_remove_module_search_path(const char* path);
int galileo_get_search_paths(char paths[][MAX_MODULE_PATH_LENGTH], int max_paths);

/* Dynamic module loading and unloading */
int galileo_load_module(const char* name);
int galileo_unload_module(const char* name);
int galileo_reload_module(const char* name);

/* Module status and information queries */
int galileo_is_module_loaded(const char* name);
int galileo_is_module_available(const char* name);
int galileo_get_module_info(const char* name, ModuleLoadInfo* info);

/* Get function pointer from loaded module */
void* galileo_get_module_symbol(const char* module_name, const char* symbol_name);

/* Module discovery and listing */
int galileo_discover_modules(void);
int galileo_list_modules(ModuleLoadInfo* modules, int max_modules);
int galileo_list_loaded_modules(ModuleLoadInfo* modules, int max_modules);

/* =============================================================================
 * ADVANCED MODULE MANAGEMENT API
 * =============================================================================
 */

/* Dependency management */
int galileo_load_module_with_dependencies(const char* name);
int galileo_get_module_dependencies(const char* name, char deps[][MAX_MODULE_NAME_LENGTH], int max_deps);
int galileo_check_dependency_conflicts(const char* name);

/* Batch operations */
int galileo_load_modules_batch(const char* names[], int count);
int galileo_unload_modules_batch(const char* names[], int count);
int galileo_load_required_modules(void);
int galileo_unload_optional_modules(void);

/* Module capability queries */
int galileo_get_module_capabilities(const char* name);
int galileo_find_modules_with_capability(ModuleCapabilities capability, 
                                        char names[][MAX_MODULE_NAME_LENGTH], int max_modules);

/* =============================================================================
 * DEVELOPMENT AND DEBUGGING API
 * =============================================================================
 */

/* Hot-reload support for development */
int galileo_enable_hot_reload(const char* name);
int galileo_disable_hot_reload(const char* name);
int galileo_check_for_module_updates(void);

/* Statistics and performance monitoring */
int galileo_get_loading_stats(ModuleLoadingStats* stats);
void galileo_reset_loading_stats(void);
void galileo_print_module_report(FILE* output);

/* Module validation and integrity */
int galileo_validate_module(const char* path);
int galileo_check_module_compatibility(const char* name);
int galileo_verify_module_signatures(void);

/* =============================================================================
 * CALLBACK AND EVENT SYSTEM
 * =============================================================================
 */

/* Module loading event types */
typedef enum {
    MODULE_EVENT_DISCOVERED,                   /* Module discovered during scan */
    MODULE_EVENT_LOAD_START,                   /* Module loading started */
    MODULE_EVENT_LOAD_SUCCESS,                 /* Module loaded successfully */
    MODULE_EVENT_LOAD_FAILED,                  /* Module loading failed */
    MODULE_EVENT_UNLOAD_START,                 /* Module unloading started */
    MODULE_EVENT_UNLOAD_SUCCESS,               /* Module unloaded successfully */
    MODULE_EVENT_DEPENDENCY_RESOLVED,          /* Dependency resolved */
    MODULE_EVENT_HOT_RELOAD                    /* Module hot-reloaded */
} ModuleEventType;

/* Module event callback function type */
typedef void (*ModuleEventCallback)(ModuleEventType event, const char* module_name, void* user_data);

/* Event callback management */
int galileo_register_module_event_callback(ModuleEventCallback callback, void* user_data);
int galileo_unregister_module_event_callback(ModuleEventCallback callback);

/* =============================================================================
 * UTILITY FUNCTIONS AND HELPERS
 * =============================================================================
 */

/* Module name and path utilities */
int galileo_module_name_from_path(const char* path, char* name, size_t name_size);
int galileo_module_path_from_name(const char* name, char* path, size_t path_size);
int galileo_is_valid_module_name(const char* name);

/* String manipulation helpers for module names */
int galileo_normalize_module_name(char* name);
int galileo_compare_module_versions(const char* version1, const char* version2);

/* File system utilities */
int galileo_file_exists(const char* path);
long galileo_get_file_modification_time(const char* path);
int galileo_create_directory_if_needed(const char* path);

/* =============================================================================
 * ERROR HANDLING AND RETURN CODES
 * =============================================================================
 */

/* Return codes for module loading operations */
#define MODULE_LOAD_SUCCESS                 0   /* Operation completed successfully */
#define MODULE_LOAD_ERROR_NOT_FOUND        -1   /* Module not found */
#define MODULE_LOAD_ERROR_ALREADY_LOADED   -2   /* Module already loaded */
#define MODULE_LOAD_ERROR_INVALID_PATH     -3   /* Invalid module path */
#define MODULE_LOAD_ERROR_DLOPEN_FAILED    -4   /* dlopen() failed */
#define MODULE_LOAD_ERROR_SYMBOL_MISSING   -5   /* Required symbol not found */
#define MODULE_LOAD_ERROR_INIT_FAILED      -6   /* Module initialization failed */
#define MODULE_LOAD_ERROR_INCOMPATIBLE     -7   /* Module incompatible with system */
#define MODULE_LOAD_ERROR_DEPENDENCY       -8   /* Dependency error */
#define MODULE_LOAD_ERROR_MEMORY           -9   /* Memory allocation failed */
#define MODULE_LOAD_ERROR_PERMISSION       -10  /* Permission denied */
#define MODULE_LOAD_ERROR_SYSTEM           -11  /* System error */

/* Error message retrieval */
const char* galileo_module_error_string(int error_code);
int galileo_get_last_module_error(char* buffer, size_t buffer_size);

/* =============================================================================
 * THREAD SAFETY AND CONCURRENCY
 * =============================================================================
 */

/* Thread-safe operation modes */
typedef enum {
    MODULE_THREAD_MODE_SINGLE,                  /* Single-threaded mode (no locking) */
    MODULE_THREAD_MODE_MULTI,                   /* Multi-threaded mode (with locking) */
    MODULE_THREAD_MODE_LOCKFREE                 /* Lock-free mode (experimental) */
} ModuleThreadMode;

/* Thread safety configuration */
int galileo_set_module_thread_mode(ModuleThreadMode mode);
ModuleThreadMode galileo_get_module_thread_mode(void);

/* Manual locking for custom synchronization */
int galileo_module_registry_lock(void);
int galileo_module_registry_unlock(void);

/* =============================================================================
 * CONFIGURATION AND ENVIRONMENT
 * =============================================================================
 */

/* Environment variable support */
#define GALILEO_MODULE_PATH_ENV "GALILEO_MODULE_PATH"       /* Override module search paths */
#define GALILEO_MODULE_DEBUG_ENV "GALILEO_MODULE_DEBUG"     /* Enable debug output */
#define GALILEO_MODULE_VERBOSE_ENV "GALILEO_MODULE_VERBOSE" /* Enable verbose logging */

/* Configuration loading and saving */
int galileo_load_module_config(const char* config_file);
int galileo_save_module_config(const char* config_file);

/* Runtime configuration queries */
int galileo_get_module_loader_version(char* version, size_t version_size);
int galileo_is_module_loader_initialized(void);

/* =============================================================================
 * FORWARD COMPATIBILITY AND EXTENSIONS
 * =============================================================================
 */

/* Plugin interface for extending the module loader */
typedef struct {
    const char* name;                           /* Extension name */
    int version;                                /* Extension version */
    int (*init)(void);                          /* Extension initialization */
    void (*cleanup)(void);                      /* Extension cleanup */
    int (*load_module)(const char* name);       /* Custom module loading */
    int (*validate_module)(const char* path);   /* Custom module validation */
} ModuleLoaderExtension;

/* Extension management */
int galileo_register_loader_extension(const ModuleLoaderExtension* extension);
int galileo_unregister_loader_extension(const char* name);

/* Future-proofing: opaque handle for advanced features */
typedef struct ModuleHandle ModuleHandle;

/* Advanced module handle operations (for future expansion) */
ModuleHandle* galileo_get_module_handle(const char* name);
int galileo_release_module_handle(ModuleHandle* handle);

#endif /* GALILEO_MODULE_LOADER_H */
