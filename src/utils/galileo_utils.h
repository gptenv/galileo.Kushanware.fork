/* =============================================================================
 * galileo/src/utils/galileo_utils.h - Utilities Module Public API
 * 
 * Public header for the utilities module containing all functions for
 * tokenization, file I/O, output formatting, and utility functions that
 * make Galileo practical for real-world use.
 * 
 * This module provides all the practical utilities needed to interface
 * with the outside world - input processing, output formatting, and
 * helper functions for a production-ready system.
 * =============================================================================
 */

#ifndef GALILEO_UTILS_H
#define GALILEO_UTILS_H

#include "../core/galileo_types.h"
#include <stdio.h>
#include <stddef.h>

/* =============================================================================
 * TOKENIZATION AND INPUT PROCESSING
 * =============================================================================
 */

/* Enhanced tokenization with smart word boundary detection */
char** tokenize_input(const char* input, int* token_count);

/* Free token array and associated memory */
void free_tokens(char** tokens, int count);

/* =============================================================================
 * STDIN AND FILE INPUT PROCESSING
 * =============================================================================
 */

/* Check if stdin has data available (for auto-detection) */
int is_stdin_available(void);

/* Read all input from stdin */
char* read_stdin_input(void);

/* Read file content into string */
char* read_file_content(const char* filename);

/* Process file input through Galileo model */
int process_file_input(GalileoModel* model, const char* filename);

/* =============================================================================
 * OUTPUT FORMATTING AND DISPLAY
 * =============================================================================
 */

/* Print comprehensive model summary */
void print_model_summary(GalileoModel* model, FILE* output);

/* Print all learned facts in a beautiful format */
void print_facts(GalileoModel* model, FILE* output);

/* Print graph statistics in detailed format */
void print_graph_stats(GalileoModel* model, FILE* output);

/* =============================================================================
 * ADVANCED OUTPUT FORMATTING
 * =============================================================================
 */

/* Print memory system status and statistics */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
void print_memory_status(GalileoModel* model, FILE* output);
#endif

/* Print symbolic reasoning results and conflicts */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
void print_symbolic_analysis(GalileoModel* model, FILE* output);
#endif

/* Print detailed node information */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
void print_node_details(GalileoModel* model, int node_index, FILE* output);
#endif

/* Print edge connectivity analysis */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
void print_edge_analysis(GalileoModel* model, FILE* output);
#endif

/* =============================================================================
 * TOKENIZATION CONFIGURATION
 * All constants can be overridden at compile time
 * =============================================================================
 */

/* Tokenization parameters */
#ifndef MAX_ESTIMATED_TOKENS_RATIO
#define MAX_ESTIMATED_TOKENS_RATIO 2
#endif

#ifndef TOKEN_BUFFER_SAFETY_MARGIN
#define TOKEN_BUFFER_SAFETY_MARGIN 10
#endif

/* Input processing limits */
#ifndef MAX_INPUT_BUFFER_SIZE
#define MAX_INPUT_BUFFER_SIZE (16 * 1024 * 1024)  /* 16MB */
#endif

#ifndef INITIAL_INPUT_BUFFER_SIZE
#define INITIAL_INPUT_BUFFER_SIZE 4096
#endif

#ifndef INPUT_BUFFER_GROWTH_FACTOR
#define INPUT_BUFFER_GROWTH_FACTOR 2
#endif

/* File processing parameters */
#ifndef MAX_FILE_SIZE
#define MAX_FILE_SIZE (100 * 1024 * 1024)  /* 100MB */
#endif

#ifndef FILE_READ_CHUNK_SIZE
#define FILE_READ_CHUNK_SIZE 8192
#endif

/* Output formatting parameters */
#ifndef CONFIDENCE_BAR_LENGTH
#define CONFIDENCE_BAR_LENGTH 10
#endif

#ifndef MAX_FACTS_DISPLAY_DEFAULT
#define MAX_FACTS_DISPLAY_DEFAULT 50
#endif

#ifndef MAX_NODES_DISPLAY_DEFAULT
#define MAX_NODES_DISPLAY_DEFAULT 20
#endif

/* =============================================================================
 * TOKENIZATION UTILITY STRUCTURES
 * =============================================================================
 */

/* Token statistics for analysis */
typedef struct {
    int total_tokens;                       /* Total number of tokens */
    int unique_tokens;                      /* Number of unique tokens */
    int average_token_length;               /* Average token length */
    int longest_token_length;               /* Length of longest token */
    int shortest_token_length;              /* Length of shortest token */
    float lexical_diversity;                /* Unique tokens / total tokens */
} TokenStatistics;

/* Input processing result */
typedef struct {
    char** tokens;                          /* Array of token strings */
    int token_count;                        /* Number of tokens */
    TokenStatistics stats;                  /* Token statistics */
    int processing_time_ms;                 /* Processing time in milliseconds */
    int success;                            /* Was processing successful? */
    char error_message[256];                /* Error message if failed */
} InputProcessingResult;

/* =============================================================================
 * ADVANCED TOKENIZATION OPERATIONS
 * =============================================================================
 */

/* Enhanced tokenization with statistics */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
InputProcessingResult tokenize_input_advanced(const char* input);
#endif

/* Token normalization utilities */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
void normalize_token(char* token);
int is_separator(char c);
void compute_token_statistics(char** tokens, int count, TokenStatistics* stats);
#endif

/* =============================================================================
 * FILE AND I/O UTILITIES
 * =============================================================================
 */

/* File validation and safety checks */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
int validate_file_access(const char* filename);
int get_file_size(const char* filename);
int is_text_file(const char* filename);
#endif

/* Safe file reading with error handling */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
char* read_file_safe(const char* filename, size_t* bytes_read, char* error_buffer, size_t error_buffer_size);
#endif

/* Streaming file processing for large files */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
int process_large_file_streaming(GalileoModel* model, const char* filename, 
                                int (*chunk_processor)(GalileoModel*, const char*, void*), 
                                void* user_data);
#endif

/* =============================================================================
 * OUTPUT FORMATTING UTILITIES
 * =============================================================================
 */

/* Formatting helper structures */
typedef struct {
    int show_confidence_bars;               /* Show visual confidence indicators */
    int show_supporting_nodes;              /* Show supporting node information */
    int show_timestamps;                    /* Show iteration timestamps */
    int show_detailed_stats;                /* Show detailed statistics */
    int max_facts_to_display;               /* Maximum facts to display */
    int max_nodes_to_display;               /* Maximum nodes to display */
    int use_colors;                         /* Use ANSI color codes */
    int compact_format;                     /* Use compact formatting */
} OutputFormatOptions;

/* Default output format options */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
extern const OutputFormatOptions DEFAULT_OUTPUT_OPTIONS;
#endif

/* Advanced output formatting with options */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
void print_model_summary_formatted(GalileoModel* model, FILE* output, const OutputFormatOptions* options);
void print_facts_formatted(GalileoModel* model, FILE* output, const OutputFormatOptions* options);
void print_graph_stats_formatted(GalileoModel* model, FILE* output, const OutputFormatOptions* options);
#endif

/* =============================================================================
 * STRING AND TEXT UTILITIES
 * =============================================================================
 */

/* String processing utilities */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
char* trim_whitespace(char* str);
char* string_duplicate(const char* str);
int string_starts_with(const char* str, const char* prefix);
int string_ends_with(const char* str, const char* suffix);
void string_to_lowercase(char* str);
void string_replace_char(char* str, char old_char, char new_char);
#endif

/* Text analysis utilities */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
int count_words(const char* text);
int count_sentences(const char* text);
int count_characters(const char* text, int include_whitespace);
float calculate_readability_score(const char* text);
#endif

/* =============================================================================
 * PERFORMANCE AND TIMING UTILITIES
 * =============================================================================
 */

/* Timing and performance measurement */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
typedef struct {
    long start_time_ms;
    long end_time_ms;
    long duration_ms;
    const char* operation_name;
} PerformanceTimer;

void start_timer(PerformanceTimer* timer, const char* operation_name);
void stop_timer(PerformanceTimer* timer);
void print_timing_report(const PerformanceTimer* timer, FILE* output);
#endif

/* Memory usage utilities */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
size_t get_memory_usage_bytes(void);
void print_memory_usage_report(FILE* output);
#endif

/* =============================================================================
 * ERROR HANDLING AND LOGGING
 * =============================================================================
 */

/* Error handling utilities */
#ifdef GALILEO_EXPOSE_ADVANCED_UTILS
typedef enum {
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL
} LogLevel;

void log_message(LogLevel level, const char* format, ...);
void set_log_level(LogLevel level);
void enable_log_timestamps(int enable);
void set_log_output_file(FILE* output);
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
} UtilsModuleInfo;

extern UtilsModuleInfo utils_module_info;

#endif /* GALILEO_UTILS_H */
