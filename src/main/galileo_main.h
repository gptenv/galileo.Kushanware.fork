/* =============================================================================
 * galileo/src/main/galileo_main.h - CLI Interface Module Public API
 * 
 * Public header for the main CLI interface module containing all functions
 * for command-line argument parsing, interactive mode, file processing,
 * and program orchestration.
 * 
 * This module provides the user-facing interface that ties all other modules
 * together into a cohesive, professional command-line application.
 * =============================================================================
 */

#ifndef GALILEO_MAIN_H
#define GALILEO_MAIN_H

#include "../core/galileo_types.h"
#include <stdio.h>

/* =============================================================================
 * COMMAND LINE OPTION STRUCTURE
 * =============================================================================
 */

/* Comprehensive command line options structure */
typedef struct {
    /* Basic operation modes */
    int verbose;                            /* -v, --verbose */
    int quiet;                              /* -q, --quiet */
    int help;                               /* -h, --help */
    int version;                            /* -V, --version */
    int test_mode;                          /* -t, --test */
    int interactive;                        /* -i, --interactive */
    
    /* Input/output options */
    char* output_file;                      /* -o, --output FILE */
    char* config_file;                      /* -c, --config FILE */
    char** input_files;                     /* Remaining arguments */
    int input_file_count;                   /* Number of input files */
    
    /* Processing parameters */
    int max_iterations;                     /* --max-iterations N */
    float similarity_threshold;             /* --similarity-threshold F */
    float attention_threshold;              /* --attention-threshold F */
    
    /* Module control */
    int disable_symbolic;                   /* --no-symbolic */
    int disable_compression;                /* --no-compression */
    int disable_graph;                      /* --no-graph */
    int disable_memory;                     /* --no-memory */
    
    /* Advanced options */
    int list_modules;                       /* --list-modules */
    int benchmark_mode;                     /* --benchmark */
    int debug_mode;                         /* --debug */
    int profile_mode;                       /* --profile */
    
    /* Output formatting */
    int show_stats;                         /* --show-stats */
    int show_facts;                         /* --show-facts */
    int show_graph;                         /* --show-graph */
    int show_memory;                        /* --show-memory */
    int compact_output;                     /* --compact */
    int json_output;                        /* --json */
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

/* Configuration management */
int load_config_file(const char* config_file, GalileoOptions* options);
int save_config_file(const char* config_file, const GalileoOptions* options);

/* Processing functions */
int process_input_text(GalileoModel* model, const char* text, const GalileoOptions* options);
int process_stdin(GalileoModel* model, const GalileoOptions* options);
int process_files(GalileoModel* model, const GalileoOptions* options);

/* Interactive mode */
int run_interactive_mode(GalileoModel* model, const GalileoOptions* options);

/* Main entry point */
int main(int argc, char* argv[]);

/* =============================================================================
 * INTERACTIVE MODE STRUCTURES AND FUNCTIONS
 * =============================================================================
 */

/* Interactive command structure */
typedef struct {
    const char* command;                    /* Command name */
    const char* description;                /* Command description */
    int (*handler)(GalileoModel* model, const char* args, const GalileoOptions* options);
} InteractiveCommand;

/* Interactive mode functions */
#ifdef GALILEO_EXPOSE_INTERACTIVE_INTERNALS
int handle_help_command(GalileoModel* model, const char* args, const GalileoOptions* options);
int handle_stats_command(GalileoModel* model, const char* args, const GalileoOptions* options);
int handle_facts_command(GalileoModel* model, const char* args, const GalileoOptions* options);
int handle_clear_command(GalileoModel* model, const char* args, const GalileoOptions* options);
int handle_save_command(GalileoModel* model, const char* args, const GalileoOptions* options);
int handle_load_command(GalileoModel* model, const char* args, const GalileoOptions* options);
int handle_debug_command(GalileoModel* model, const char* args, const GalileoOptions* options);
int handle_modules_command(GalileoModel* model, const char* args, const GalileoOptions* options);
int handle_benchmark_command(GalileoModel* model, const char* args, const GalileoOptions* options);
#endif

/* =============================================================================
 * MODULE MANAGEMENT
 * =============================================================================
 */

/* Module loading and management */
int load_required_modules(void);
int load_optional_modules(const GalileoOptions* options);
void unload_all_modules(void);
int is_module_loaded(const char* module_name);
void print_module_status(FILE* output);

/* Module registry management */
#ifdef GALILEO_EXPOSE_MODULE_INTERNALS
typedef struct {
    void* handle;                           /* dlopen handle */
    const char* name;                       /* Module name */
    const char* path;                       /* Module file path */
    int loaded;                             /* Is module loaded? */
    int required;                           /* Is module required? */
    int (*init_func)(void);                /* Module initialization function */
    void (*cleanup_func)(void);            /* Module cleanup function */
} ModuleRegistry;

extern ModuleRegistry g_modules[];
extern int g_module_count;

int register_module(const char* name, const char* path, int required);
int load_module_by_name(const char* name);
void unload_module_by_name(const char* name);
#endif

/* =============================================================================
 * SIGNAL HANDLING AND CLEANUP
 * =============================================================================
 */

/* Signal handling for graceful shutdown */
void setup_signal_handlers(void);
void signal_handler(int signal);
void cleanup_and_exit(int exit_code);

/* Global state management */
extern volatile int g_shutdown_requested;
extern GalileoModel* g_current_model;

/* =============================================================================
 * TESTING AND BENCHMARKING
 * =============================================================================
 */

/* Built-in test suites */
int run_unit_tests(const GalileoOptions* options);
int run_integration_tests(const GalileoOptions* options);
int run_performance_tests(const GalileoOptions* options);

/* Benchmarking functions */
#ifdef GALILEO_EXPOSE_BENCHMARK_INTERNALS
typedef struct {
    const char* test_name;
    int (*test_function)(GalileoModel* model);
    int iterations;
    double average_time_ms;
    double min_time_ms;
    double max_time_ms;
    int success_count;
    int failure_count;
} BenchmarkResult;

int benchmark_tokenization(GalileoModel* model);
int benchmark_graph_operations(GalileoModel* model);
int benchmark_symbolic_reasoning(GalileoModel* model);
int benchmark_memory_operations(GalileoModel* model);
void print_benchmark_results(const BenchmarkResult* results, int count, FILE* output);
#endif

/* =============================================================================
 * CONFIGURATION CONSTANTS
 * All constants can be overridden at compile time
 * =============================================================================
 */

/* Interactive mode parameters */
#ifndef MAX_INTERACTIVE_COMMAND_LENGTH
#define MAX_INTERACTIVE_COMMAND_LENGTH 1024
#endif

#ifndef MAX_INTERACTIVE_HISTORY
#define MAX_INTERACTIVE_HISTORY 100
#endif

#ifndef INTERACTIVE_PROMPT_FORMAT
#define INTERACTIVE_PROMPT_FORMAT "galileo[%d]> "
#endif

/* Module loading parameters */
#ifndef MAX_MODULE_PATH_LENGTH
#define MAX_MODULE_PATH_LENGTH 512
#endif

#ifndef DEFAULT_MODULE_DIRECTORY
#define DEFAULT_MODULE_DIRECTORY "./lib"
#endif

#ifndef MODULE_FILE_EXTENSION
#define MODULE_FILE_EXTENSION ".so"
#endif

/* Configuration file parameters */
#ifndef MAX_CONFIG_LINE_LENGTH
#define MAX_CONFIG_LINE_LENGTH 256
#endif

#ifndef DEFAULT_CONFIG_FILE
#define DEFAULT_CONFIG_FILE "galileo.conf"
#endif

/* Output formatting parameters */
#ifndef DEFAULT_OUTPUT_WIDTH
#define DEFAULT_OUTPUT_WIDTH 80
#endif

#ifndef PROGRESS_BAR_WIDTH
#define PROGRESS_BAR_WIDTH 20
#endif

/* Performance and timing parameters */
#ifndef BENCHMARK_DEFAULT_ITERATIONS
#define BENCHMARK_DEFAULT_ITERATIONS 10
#endif

#ifndef PERFORMANCE_SAMPLE_INTERVAL_MS
#define PERFORMANCE_SAMPLE_INTERVAL_MS 1000
#endif

/* =============================================================================
 * ERROR CODES AND EXIT STATUS
 * =============================================================================
 */

/* Standard exit codes */
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
 * ADVANCED CLI FEATURES
 * =============================================================================
 */

/* Progress reporting for long operations */
#ifdef GALILEO_EXPOSE_PROGRESS_INTERNALS
typedef struct {
    const char* operation_name;
    int total_steps;
    int completed_steps;
    long start_time_ms;
    int show_percentage;
    int show_eta;
    FILE* output;
} ProgressReporter;

void init_progress_reporter(ProgressReporter* reporter, const char* operation, int total_steps, FILE* output);
void update_progress(ProgressReporter* reporter, int completed_steps);
void finish_progress(ProgressReporter* reporter);
#endif

/* Configuration validation and defaults */
#ifdef GALILEO_EXPOSE_CONFIG_INTERNALS
int validate_options(const GalileoOptions* options, char* error_buffer, size_t buffer_size);
void set_default_options(GalileoOptions* options);
void merge_options(GalileoOptions* target, const GalileoOptions* source);
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
} MainModuleInfo;

extern MainModuleInfo main_module_info;

#endif /* GALILEO_MAIN_H */
