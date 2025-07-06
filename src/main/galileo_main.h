/* =============================================================================
 * galileo/src/main/galileo_main.h - Main Module Public API
 * 
 * UPDATED for lazy loading hot-pluggable module system.
 * No hardcoded module lists - everything is discovery-based with JIT loading.
 * =============================================================================
 */

#ifndef GALILEO_MAIN_H
#define GALILEO_MAIN_H

#include "../core/galileo_types.h"

/* =============================================================================
 * COMMAND LINE OPTIONS STRUCTURE
 * =============================================================================
 */

/* Command line option structure - UPDATED with new options */
typedef struct {
    /* Basic options */
    int verbose;                    /* -v, --verbose */
    int quiet;                      /* -q, --quiet */
    int help;                       /* -h, --help (1 = normal, 2 = list-modules) */
    int version;                    /* --version */
    int test_mode;                  /* -t, --test */
    int interactive;                /* -i, --interactive */
    
    /* File options */
    char* output_file;              /* -o, --output FILE */
    char* config_file;              /* -c, --config FILE */
    char** input_files;             /* Remaining arguments */
    int input_file_count;           /* Number of input files */
    
    /* Model parameters */
    int max_iterations;             /* --max-iterations N */
    float similarity_threshold;     /* --similarity-threshold F */
    float attention_threshold;      /* --attention-threshold F */
    
    /* Module disable flags - NEW! */
    int disable_symbolic;           /* --no-symbolic */
    int disable_compression;        /* --no-compression */
    int disable_memory;             /* --no-memory */
    int disable_graph;              /* --no-graph */
} GalileoOptions;

/* =============================================================================
 * CLI FUNCTION DECLARATIONS
 * =============================================================================
 */

/* Command line parsing and help */
void print_usage(const char* program_name);
void print_version(void);
int parse_arguments(int argc, char* argv[], GalileoOptions* options);
void cleanup_options(GalileoOptions* options);

/* Processing functions */
int process_input_text(GalileoModel* model, const char* text, const GalileoOptions* options);
int process_stdin(GalileoModel* model, const GalileoOptions* options);
int process_files(GalileoModel* model, const GalileoOptions* options);

/* Mode functions */
int run_interactive_mode(GalileoModel* model, const GalileoOptions* options);
int run_test_suite(GalileoModel* model, const GalileoOptions* options);

/* Main entry point */
int main(int argc, char* argv[]);

/* =============================================================================
 * LAZY MODULE LOADING FUNCTIONS - NEW!
 * =============================================================================
 */

/* Bootstrap and lazy loading */
int load_bootstrap_modules(void);
int ensure_module_loaded(const char* module_name);
int load_all_discovered_modules(void);

/* Module status and management */
int is_module_loaded(const char* module_name);
void print_module_status(FILE* output);

/* Input utilities */
int is_stdin_available(void);
char* read_stdin_input(void);

/* =============================================================================
 * SIGNAL HANDLING AND CLEANUP
 * =============================================================================
 */

/* Signal handling */
void signal_handler(int signal);
void setup_signal_handlers(void);
void cleanup_and_exit(int exit_code);

/* =============================================================================
 * EXIT CODES
 * =============================================================================
 */

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif

#ifndef EXIT_GENERAL_ERROR
#define EXIT_GENERAL_ERROR 1
#endif

#ifndef EXIT_ARGUMENT_ERROR
#define EXIT_ARGUMENT_ERROR 2
#endif

#ifndef EXIT_MODULE_ERROR
#define EXIT_MODULE_ERROR 3
#endif

#ifndef EXIT_IO_ERROR
#define EXIT_IO_ERROR 4
#endif

#ifndef EXIT_MEMORY_ERROR
#define EXIT_MEMORY_ERROR 5
#endif

#ifndef EXIT_SIGNAL_INTERRUPTED
#define EXIT_SIGNAL_INTERRUPTED 130
#endif

/* =============================================================================
 * MODULE INFO FOR DYNAMIC LOADING
 * =============================================================================
 */

/* Module info structure for dynamic loading */
typedef struct {
    const char* name;
    const char* version;
    int (*init_func)(void);
    void (*cleanup_func)(void);
} MainModuleInfo;

extern MainModuleInfo main_module_info;

/* =============================================================================
 * COMPATIBILITY AND FORWARD DECLARATIONS
 * =============================================================================
 */

/* Global state - used for signal handling */
extern volatile int g_shutdown_requested;
extern GalileoModel* g_current_model;

#endif /* GALILEO_MAIN_H */
