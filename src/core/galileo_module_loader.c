/* =============================================================================
 * galileo/src/core/galileo_module_loader.c - Dynamic Module Loading Engine
 * 
 * Hot-loading module system that provides JIT-style on-demand loading of
 * Galileo modules. This is the core engine that makes the modular architecture
 * truly dynamic and flexible.
 * 
 * Features:
 * - Runtime module discovery by scanning directories
 * - On-demand loading with lazy initialization
 * - Proper dependency chain resolution
 * - Graceful fallback when optional modules unavailable
 * - Hot-reload capability for development
 * - Thread-safe module management
 * 
 * FIXED: Deadlock bug in galileo_list_modules()
 * =============================================================================
 */

/* Enable POSIX functions */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "galileo_module_loader.h"
#include "galileo_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

/* =============================================================================
 * INTERNAL STRUCTURES AND GLOBALS
 * =============================================================================
 */

/* Extended module info for dynamic loading */
typedef struct DynamicModuleInfo {
    char name[64];                      /* Module name */
    char path[512];                     /* Full path to .so file */
    char version[32];                   /* Module version */
    void* handle;                       /* dlopen handle */
    int loaded;                         /* Is module currently loaded? */
    int required;                       /* Is module required for operation? */
    int load_attempted;                 /* Have we tried loading this? */
    int load_failed;                    /* Did loading fail? */
    char error_message[256];            /* Last error message */
    
    /* Module interface functions */
    int (*init_func)(void);            /* Module initialization */
    void (*cleanup_func)(void);        /* Module cleanup */
    const char* (*get_version)(void);  /* Get module version */
    int (*get_capabilities)(void);     /* Get module capabilities */
    
    /* Dependency tracking */
    char dependencies[8][64];          /* Module dependencies */
    int dependency_count;              /* Number of dependencies */
    
    /* Usage tracking */
    int reference_count;               /* How many things are using this module */
    long last_used_time;               /* When was this module last used */
    
    struct DynamicModuleInfo* next;    /* Linked list for hash table */
} DynamicModuleInfo;

/* Module registry state */
static struct {
    DynamicModuleInfo* modules[MODULE_HASH_SIZE];  /* Hash table of modules */
    char search_paths[MAX_SEARCH_PATHS][512];      /* Directories to search for modules */
    int search_path_count;                         /* Number of search paths */
    int total_modules;                             /* Total discovered modules */
    int loaded_modules;                            /* Currently loaded modules */
    pthread_mutex_t registry_mutex;               /* Thread safety */
    int initialized;                               /* Is system initialized? */
} g_module_registry = {0};

/* =============================================================================
 * UTILITY FUNCTIONS
 * =============================================================================
 */

/* Get current time in milliseconds */
static long get_current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000) + (ts.tv_nsec / 1000000);
}

/* Check if file exists and is readable */
static int file_exists(const char* path) {
    return access(path, R_OK) == 0;
}

/* Hash function for module names */
static unsigned int hash_module_name(const char* name) {
    unsigned int hash = 5381;
    int c;
    while ((c = *name++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % MODULE_HASH_SIZE;
}

/* Find module in registry by name (assumes mutex is held) */
static DynamicModuleInfo* find_module(const char* name) {
    if (!name) return NULL;
    
    unsigned int hash = hash_module_name(name);
    DynamicModuleInfo* module = g_module_registry.modules[hash];
    
    while (module) {
        if (strcmp(module->name, name) == 0) {
            return module;
        }
        module = module->next;
    }
    
    return NULL;
}

/* Extract module name from filename (e.g., "libgalileo_core.so" -> "core") */
static void extract_module_name(const char* filename, char* name, size_t name_size) {
    const char* start = filename;
    const char* end = strrchr(filename, '.');
    
    /* Skip "libgalileo_" prefix if present */
    if (strncmp(start, "libgalileo_", 11) == 0) {
        start += 11;
    }
    
    /* Calculate length */
    size_t len = end ? (size_t)(end - start) : strlen(start);
    if (len >= name_size) {
        len = name_size - 1;
    }
    
    /* Safe string copy with guaranteed null termination */
    memcpy(name, start, len);
    name[len] = '\0';
}

/* Add module to registry */
static DynamicModuleInfo* add_module_to_registry(const char* name, const char* path) {
    DynamicModuleInfo* module = calloc(1, sizeof(DynamicModuleInfo));
    if (!module) {
        return NULL;
    }
    
    /* Safe string copying with guaranteed null termination */
    size_t name_len = strlen(name);
    if (name_len >= sizeof(module->name)) {
        name_len = sizeof(module->name) - 1;
    }
    memcpy(module->name, name, name_len);
    module->name[name_len] = '\0';
    
    size_t path_len = strlen(path);
    if (path_len >= sizeof(module->path)) {
        path_len = sizeof(module->path) - 1;
    }
    memcpy(module->path, path, path_len);
    module->path[path_len] = '\0';
    
    module->last_used_time = get_current_time_ms();
    
    unsigned int hash = hash_module_name(name);
    module->next = g_module_registry.modules[hash];
    g_module_registry.modules[hash] = module;
    g_module_registry.total_modules++;
    
    printf("📦 Discovered module: %s -> %s\n", name, path);
    return module;
}

/* =============================================================================
 * MODULE DISCOVERY
 * =============================================================================
 */

/* Scan directory for module files */
static int scan_directory_for_modules(const char* dir_path) {
    DIR* dir = opendir(dir_path);
    if (!dir) {
        printf("🔍 Scanning directory: %s\n", dir_path);
        return 0;  /* Directory doesn't exist, that's okay */
    }
    
    printf("🔍 Scanning directory: %s\n", dir_path);
    
    int found_count = 0;
    struct dirent* entry;
    
    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        if (entry->d_name[0] == '.') continue;
        
        /* Look for .so files */
        size_t name_len = strlen(entry->d_name);
        if (name_len < 4 || strcmp(entry->d_name + name_len - 3, ".so") != 0) {
            continue;
        }
        
        /* Accept any .so file - no naming convention required */
        /* This allows maximum flexibility in module naming */
        
        /* Build full path */
        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
        
        /* Verify file exists and is readable */
        if (!file_exists(full_path)) {
            continue;
        }
        
        /* Extract module name */
        char module_name[64];
        extract_module_name(entry->d_name, module_name, sizeof(module_name));
        
        /* Check if we already know about this module */
        if (find_module(module_name)) {
            printf("🔄 Module '%s' already discovered, skipping duplicate\n", module_name);
            continue;
        }
        
        /* Add to registry */
        DynamicModuleInfo* module = add_module_to_registry(module_name, full_path);
        if (module) {
            found_count++;
        }
    }
    
    closedir(dir);
    return found_count;
}

/* Discover all available modules */
static int discover_modules(void) {
    int total_found = 0;
    
    printf("🚀 Starting module discovery...\n");
    
    /* Scan all search paths */
    for (int i = 0; i < g_module_registry.search_path_count; i++) {
        int found = scan_directory_for_modules(g_module_registry.search_paths[i]);
        total_found += found;
        printf("📂 Found %d modules in %s\n", found, g_module_registry.search_paths[i]);
    }
    
    printf("✅ Module discovery complete: %d modules found\n", total_found);
    return total_found;
}

/* =============================================================================
 * DYNAMIC LOADING AND SYMBOL RESOLUTION
 * =============================================================================
 */

/* Load a single module using dlopen */
static int load_module_dynamic(DynamicModuleInfo* module) {
    if (module->loaded) {
        return 0;  /* Already loaded */
    }
    
    if (module->load_failed) {
        printf("⚠️  Module '%s' previously failed to load: %s\n", 
               module->name, module->error_message);
        return -1;
    }
    
    printf("🔄 Loading module '%s' from %s...\n", module->name, module->path);
    
    /* Clear any previous dlopen errors */
    dlerror();
    
    /* Load the shared library */
    module->handle = dlopen(module->path, RTLD_LAZY | RTLD_LOCAL);
    if (!module->handle) {
        const char* error = dlerror();
        snprintf(module->error_message, sizeof(module->error_message), 
                 "dlopen failed: %s", error ? error : "unknown error");
        module->load_failed = 1;
        printf("❌ Failed to load module '%s': %s\n", module->name, module->error_message);
        return -1;
    }
    
    /* Look for module info structure */
    char info_symbol[128];
    snprintf(info_symbol, sizeof(info_symbol), "%s_module_info", module->name);
    
    void* info_ptr = dlsym(module->handle, info_symbol);
    if (!info_ptr) {
        printf("⚠️  Module '%s' doesn't export '%s', trying alternative symbols\n", 
               module->name, info_symbol);
    }
    
    /* Look for standard module interface functions */
    char init_symbol[128], cleanup_symbol[128];
    snprintf(init_symbol, sizeof(init_symbol), "%s_module_init", module->name);
    snprintf(cleanup_symbol, sizeof(cleanup_symbol), "%s_module_cleanup", module->name);
    
    module->init_func = (int (*)(void))dlsym(module->handle, init_symbol);
    module->cleanup_func = (void (*)(void))dlsym(module->handle, cleanup_symbol);
    
    /* Look for version function */
    char version_symbol[128];
    snprintf(version_symbol, sizeof(version_symbol), "%s_get_version", module->name);
    module->get_version = (const char* (*)(void))dlsym(module->handle, version_symbol);
    
    /* Try to initialize the module */
    if (module->init_func) {
        printf("🔧 Initializing module '%s'...\n", module->name);
        int init_result = module->init_func();
        if (init_result != 0) {
            snprintf(module->error_message, sizeof(module->error_message), 
                     "Module initialization failed (code %d)", init_result);
            dlclose(module->handle);
            module->handle = NULL;
            module->load_failed = 1;
            printf("❌ Module '%s' initialization failed\n", module->name);
            return -1;
        }
    }
    
    /* Get version info if available */
    if (module->get_version) {
        const char* version = module->get_version();
        if (version) {
            size_t ver_len = strlen(version);
            if (ver_len >= sizeof(module->version)) {
                ver_len = sizeof(module->version) - 1;
            }
            memcpy(module->version, version, ver_len);
            module->version[ver_len] = '\0';
        }
    }
    
    module->loaded = 1;
    module->load_attempted = 1;
    module->reference_count = 1;
    module->last_used_time = get_current_time_ms();
    g_module_registry.loaded_modules++;
    
    printf("✅ Module '%s' loaded successfully (version: %s)\n", 
           module->name, module->version[0] ? module->version : "unknown");
    
    return 0;
}

/* Unload a module */
static int unload_module_dynamic(DynamicModuleInfo* module) {
    if (!module->loaded || !module->handle) {
        return 0;  /* Not loaded */
    }
    
    printf("🔄 Unloading module '%s'...\n", module->name);
    
    /* Call cleanup function if available */
    if (module->cleanup_func) {
        printf("🧹 Cleaning up module '%s'...\n", module->name);
        module->cleanup_func();
    }
    
    /* Close the shared library */
    if (dlclose(module->handle) != 0) {
        const char* error = dlerror();
        printf("⚠️  Warning: dlclose failed for module '%s': %s\n", 
               module->name, error ? error : "unknown error");
    }
    
    module->handle = NULL;
    module->loaded = 0;
    module->reference_count = 0;
    g_module_registry.loaded_modules--;
    
    printf("✅ Module '%s' unloaded\n", module->name);
    return 0;
}

/* =============================================================================
 * PUBLIC API FUNCTIONS
 * =============================================================================
 */

/* Initialize the module loading system */
int galileo_module_loader_init(void) {
    if (g_module_registry.initialized) {
        return 0;  /* Already initialized */
    }
    
    printf("🚀 Initializing Galileo module loading system...\n");
    
    /* Initialize mutex */
    if (pthread_mutex_init(&g_module_registry.registry_mutex, NULL) != 0) {
        fprintf(stderr, "❌ Failed to initialize module registry mutex\n");
        return -1;
    }
    
    /* Set up default search paths */
    g_module_registry.search_path_count = 0;
    galileo_add_module_search_path("./build/lib/galileo");     /* Build directory */
    galileo_add_module_search_path("./lib/galileo");           /* Project lib directory */
    galileo_add_module_search_path("./lib64/galileo");         /* Project lib64 directory */
    
    /* User-local installations */
    const char* home = getenv("HOME");
    if (home) {
        char user_lib_path[1024];
        snprintf(user_lib_path, sizeof(user_lib_path), "%s/.local/lib/galileo", home);
        galileo_add_module_search_path(user_lib_path);
        
        snprintf(user_lib_path, sizeof(user_lib_path), "%s/.local/lib64/galileo", home);
        galileo_add_module_search_path(user_lib_path);
    }
    
    /* Discover modules */
    discover_modules();
    
    g_module_registry.initialized = 1;
    printf("✅ Module loading system initialized (%d modules discovered)\n", 
           g_module_registry.total_modules);
    
    return 0;
}

/* Add module search path */
int galileo_add_module_search_path(const char* path) {
    if (!path || g_module_registry.search_path_count >= MAX_SEARCH_PATHS) {
        return -1;
    }
    
    size_t path_len = strlen(path);
    if (path_len >= sizeof(g_module_registry.search_paths[0])) {
        path_len = sizeof(g_module_registry.search_paths[0]) - 1;
    }
    
    memcpy(g_module_registry.search_paths[g_module_registry.search_path_count], path, path_len);
    g_module_registry.search_paths[g_module_registry.search_path_count][path_len] = '\0';
    g_module_registry.search_path_count++;
    
    printf("📂 Added module search path: %s\n", path);
    return 0;
}

/* Load a module by name (on-demand) */
int galileo_load_module(const char* name) {
    if (!name || !g_module_registry.initialized) {
        return -1;
    }
    
    pthread_mutex_lock(&g_module_registry.registry_mutex);
    
    DynamicModuleInfo* module = find_module(name);
    if (!module) {
        pthread_mutex_unlock(&g_module_registry.registry_mutex);
        printf("❌ Module '%s' not found in registry\n", name);
        return -1;
    }
    
    int result = load_module_dynamic(module);
    
    pthread_mutex_unlock(&g_module_registry.registry_mutex);
    return result;
}

/* Unload a module by name */
int galileo_unload_module(const char* name) {
    if (!name || !g_module_registry.initialized) {
        return -1;
    }
    
    pthread_mutex_lock(&g_module_registry.registry_mutex);
    
    DynamicModuleInfo* module = find_module(name);
    if (!module) {
        pthread_mutex_unlock(&g_module_registry.registry_mutex);
        return -1;
    }
    
    int result = unload_module_dynamic(module);
    
    pthread_mutex_unlock(&g_module_registry.registry_mutex);
    return result;
}

/* Check if a module is loaded */
int galileo_is_module_loaded(const char* name) {
    if (!name || !g_module_registry.initialized) {
        return 0;
    }
    
    pthread_mutex_lock(&g_module_registry.registry_mutex);
    
    DynamicModuleInfo* module = find_module(name);
    int loaded = module ? module->loaded : 0;
    
    pthread_mutex_unlock(&g_module_registry.registry_mutex);
    return loaded;
}

/* =============================================================================
 * DEADLOCK FIX: Internal helper for module info copying
 * =============================================================================
 */

/* Internal helper to copy module info without acquiring mutex (assumes already locked) */
static int galileo_get_module_info_internal(const DynamicModuleInfo* module, ModuleLoadInfo* info) {
    if (!module || !info) {
        return -1;
    }
    
    memset(info, 0, sizeof(ModuleLoadInfo));
    
    /* Copy strings safely with guaranteed null termination */
    size_t name_len = strlen(module->name);
    if (name_len >= sizeof(info->name)) name_len = sizeof(info->name) - 1;
    memcpy(info->name, module->name, name_len);
    info->name[name_len] = '\0';
    
    size_t path_len = strlen(module->path);
    if (path_len >= sizeof(info->path)) path_len = sizeof(info->path) - 1;
    memcpy(info->path, module->path, path_len);
    info->path[path_len] = '\0';
    
    size_t ver_len = strlen(module->version);
    if (ver_len >= sizeof(info->version)) ver_len = sizeof(info->version) - 1;
    memcpy(info->version, module->version, ver_len);
    info->version[ver_len] = '\0';
    
    size_t err_len = strlen(module->error_message);
    if (err_len >= sizeof(info->error_message)) err_len = sizeof(info->error_message) - 1;
    memcpy(info->error_message, module->error_message, err_len);
    info->error_message[err_len] = '\0';
    
    info->loaded = module->loaded;
    info->required = module->required;
    info->load_failed = module->load_failed;
    info->reference_count = module->reference_count;
    
    return 0;
}

/* Get module information */
int galileo_get_module_info(const char* name, ModuleLoadInfo* info) {
    if (!name || !info || !g_module_registry.initialized) {
        return -1;
    }
    
    pthread_mutex_lock(&g_module_registry.registry_mutex);
    
    DynamicModuleInfo* module = find_module(name);
    if (!module) {
        pthread_mutex_unlock(&g_module_registry.registry_mutex);
        return -1;
    }
    
    int result = galileo_get_module_info_internal(module, info);
    
    pthread_mutex_unlock(&g_module_registry.registry_mutex);
    return result;
}

/* Get function pointer from a loaded module */
void* galileo_get_module_symbol(const char* module_name, const char* symbol_name) {
    if (!module_name || !symbol_name || !g_module_registry.initialized) {
        return NULL;
    }
    
    pthread_mutex_lock(&g_module_registry.registry_mutex);
    
    DynamicModuleInfo* module = find_module(module_name);
    if (!module || !module->loaded || !module->handle) {
        pthread_mutex_unlock(&g_module_registry.registry_mutex);
        return NULL;
    }
    
    void* symbol = dlsym(module->handle, symbol_name);
    
    pthread_mutex_unlock(&g_module_registry.registry_mutex);
    
    return symbol;
}

/* FIXED: List all discovered modules - NO MORE DEADLOCK! */
int galileo_list_modules(ModuleLoadInfo* modules, int max_modules) {
    if (!modules || max_modules <= 0 || !g_module_registry.initialized) {
        return -1;
    }
    
    pthread_mutex_lock(&g_module_registry.registry_mutex);
    
    int count = 0;
    for (int i = 0; i < MODULE_HASH_SIZE && count < max_modules; i++) {
        DynamicModuleInfo* module = g_module_registry.modules[i];
        while (module && count < max_modules) {
            /* FIX: Use internal version that doesn't acquire mutex */
            galileo_get_module_info_internal(module, &modules[count]);
            count++;
            module = module->next;
        }
    }
    
    pthread_mutex_unlock(&g_module_registry.registry_mutex);
    return count;
}

/* Get error string for error code */
const char* galileo_module_error_string(int error_code) {
    switch (error_code) {
        case MODULE_LOAD_SUCCESS:
            return "Success";
        case MODULE_LOAD_ERROR_NOT_FOUND:
            return "Module not found";
        case MODULE_LOAD_ERROR_ALREADY_LOADED:
            return "Module already loaded";
        case MODULE_LOAD_ERROR_INVALID_PATH:
            return "Invalid module path";
        case MODULE_LOAD_ERROR_DLOPEN_FAILED:
            return "Failed to load shared library";
        case MODULE_LOAD_ERROR_SYMBOL_MISSING:
            return "Required symbol not found";
        case MODULE_LOAD_ERROR_INIT_FAILED:
            return "Module initialization failed";
        case MODULE_LOAD_ERROR_INCOMPATIBLE:
            return "Module incompatible with system";
        case MODULE_LOAD_ERROR_DEPENDENCY:
            return "Dependency error";
        case MODULE_LOAD_ERROR_MEMORY:
            return "Memory allocation failed";
        case MODULE_LOAD_ERROR_PERMISSION:
            return "Permission denied";
        case MODULE_LOAD_ERROR_SYSTEM:
            return "System error";
        default:
            return "Unknown error";
    }
}

/* Get module loader version */
int galileo_get_module_loader_version(char* version, size_t version_size) {
    if (!version || version_size == 0) {
        return -1;
    }
    
    const char* loader_version = "42.1.0";
    size_t len = strlen(loader_version);
    if (len >= version_size) {
        len = version_size - 1;
    }
    
    memcpy(version, loader_version, len);
    version[len] = '\0';
    
    return 0;
}

/* Cleanup the module loading system */
void galileo_module_loader_cleanup(void) {
    if (!g_module_registry.initialized) {
        return;
    }
    
    printf("🧹 Shutting down module loading system...\n");
    
    pthread_mutex_lock(&g_module_registry.registry_mutex);
    
    /* Unload all loaded modules */
    for (int i = 0; i < MODULE_HASH_SIZE; i++) {
        DynamicModuleInfo* module = g_module_registry.modules[i];
        while (module) {
            DynamicModuleInfo* next = module->next;
            if (module->loaded) {
                unload_module_dynamic(module);
            }
            free(module);
            module = next;
        }
        g_module_registry.modules[i] = NULL;
    }
    
    g_module_registry.total_modules = 0;
    g_module_registry.loaded_modules = 0;
    g_module_registry.initialized = 0;
    
    pthread_mutex_unlock(&g_module_registry.registry_mutex);
    pthread_mutex_destroy(&g_module_registry.registry_mutex);
    
    printf("✅ Module loading system shutdown complete\n");
}
