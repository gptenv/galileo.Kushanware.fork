/* =============================================================================
 * galileo/src/utils/galileo_utils.c - Utilities & I/O Module
 * 
 * Hot-loadable shared library implementing tokenization, file I/O, output
 * formatting, and utility functions that make Galileo practical for real-world use.
 * 
 * This is the final piece that ties everything together - the practical utilities
 * that handle input processing, output formatting, and all the helper functions
 * needed for a production-ready system.
 * 
 * Extracted from galileo_legacy_core-v42-v3.pre-modular.best.c with enhanced
 * error handling, robust tokenization, and beautiful output formatting.
 * =============================================================================
 */

#include "galileo_utils.h"
#include "../core/galileo_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>

/* =============================================================================
 * MODULE METADATA AND INITIALIZATION
 * =============================================================================
 */

static int utils_module_initialized = 0;

/* Module initialization */
static int utils_module_init(void) {
    if (utils_module_initialized) {
        return 0;  /* Already initialized */
    }
    
    fprintf(stderr, "🛠️  Utils module v42.1 initializing...\n");
    
    /* Initialize any utility subsystems */
    
    utils_module_initialized = 1;
    fprintf(stderr, "✅ Utils module ready! I/O & formatting online.\n");
    return 0;
}

/* Module cleanup */
static void utils_module_cleanup(void) {
    if (!utils_module_initialized) {
        return;
    }
    
    fprintf(stderr, "🛠️  Utils module shutting down...\n");
    utils_module_initialized = 0;
}

/* Module info structure for dynamic loading */
UtilsModuleInfo utils_module_info = {
    .name = "utils",
    .version = "42.1",
    .init_func = utils_module_init,
    .cleanup_func = utils_module_cleanup
};

/* =============================================================================
 * TOKENIZATION AND INPUT PROCESSING
 * =============================================================================
 */

/* Check if character is a word separator */
static int is_separator(char c) {
    return isspace(c) || c == '.' || c == ',' || c == ';' || c == ':' || 
           c == '!' || c == '?' || c == '"' || c == '\'' || c == '(' || 
           c == ')' || c == '[' || c == ']' || c == '{' || c == '}' ||
           c == '\n' || c == '\r' || c == '\t';
}

/* Normalize token (lowercase, remove punctuation) */
static void normalize_token(char* token) {
    if (!token) return;
    
    int len = strlen(token);
    int write_pos = 0;
    
    for (int i = 0; i < len; i++) {
        char c = token[i];
        
        /* Convert to lowercase */
        if (isalpha(c)) {
            token[write_pos++] = tolower(c);
        }
        /* Keep digits */
        else if (isdigit(c)) {
            token[write_pos++] = c;
        }
        /* Keep some punctuation that might be meaningful */
        else if (c == '-' || c == '_') {
            token[write_pos++] = c;
        }
        /* Skip other punctuation */
    }
    
    token[write_pos] = '\0';
    
    /* Remove empty tokens */
    if (write_pos == 0) {
        strcpy(token, "");
    }
}

/* Enhanced tokenization with smart word boundary detection */
char** tokenize_input(const char* input, int* token_count) {
    if (!input || !token_count) {
        if (token_count) *token_count = 0;
        return NULL;
    }
    
    /* Ensure utils module is initialized */
    if (!utils_module_initialized) {
        utils_module_init();
    }
    
    *token_count = 0;
    
    int len = strlen(input);
    if (len == 0) {
        return NULL;
    }
    
    /* Allocate token array (estimate max tokens) */
    int max_tokens = len / 2 + 10;  /* Conservative estimate */
    char** tokens = malloc(max_tokens * sizeof(char*));
    if (!tokens) {
        fprintf(stderr, "❌ Failed to allocate memory for tokens\n");
        return NULL;
    }
    
    /* Allocate working buffer */
    char* buffer = malloc(len + 1);
    if (!buffer) {
        free(tokens);
        fprintf(stderr, "❌ Failed to allocate working buffer\n");
        return NULL;
    }
    strcpy(buffer, input);
    
    /* Tokenize */
    int i = 0;
    while (i < len && *token_count < max_tokens - 1) {
        /* Skip separators */
        while (i < len && is_separator(buffer[i])) {
            i++;
        }
        
        if (i >= len) break;
        
        /* Find end of token */
        int start = i;
        while (i < len && !is_separator(buffer[i])) {
            i++;
        }
        
        /* Extract token */
        int token_len = i - start;
        if (token_len > 0 && token_len < MAX_TOKEN_LEN) {
            char* token = malloc(MAX_TOKEN_LEN);
            if (!token) {
                fprintf(stderr, "⚠️  Failed to allocate token memory\n");
                break;
            }
            
            strncpy(token, &buffer[start], token_len);
            token[token_len] = '\0';
            
            /* Normalize the token */
            normalize_token(token);
            
            /* Skip empty tokens after normalization */
            if (strlen(token) > 0) {
                tokens[*token_count] = token;
                (*token_count)++;
            } else {
                free(token);
            }
        }
    }
    
    free(buffer);
    
    /* Shrink token array to actual size */
    if (*token_count > 0) {
        char** final_tokens = realloc(tokens, *token_count * sizeof(char*));
        if (final_tokens) {
            tokens = final_tokens;
        }
    } else {
        free(tokens);
        tokens = NULL;
    }
    
    printf("🔤 Tokenized input into %d tokens\n", *token_count);
    
    return tokens;
}

/* Free token array */
void free_tokens(char** tokens, int count) {
    if (!tokens) return;
    
    for (int i = 0; i < count; i++) {
        if (tokens[i]) {
            free(tokens[i]);
        }
    }
    free(tokens);
}

/* =============================================================================
 * STDIN AND FILE INPUT PROCESSING
 * =============================================================================
 */

/* Check if stdin has data available */
int is_stdin_available(void) {
    return !isatty(STDIN_FILENO);
}

/* Read all input from stdin */
char* read_stdin_input(void) {
    if (isatty(STDIN_FILENO)) {
        /* No piped input */
        return NULL;
    }
    
    size_t buffer_size = 4096;
    size_t total_size = 0;
    char* buffer = malloc(buffer_size);
    
    if (!buffer) {
        fprintf(stderr, "❌ Failed to allocate input buffer\n");
        return NULL;
    }
    
    /* Read chunks from stdin */
    while (1) {
        size_t bytes_read = fread(buffer + total_size, 1, buffer_size - total_size - 1, stdin);
        total_size += bytes_read;
        
        if (bytes_read == 0) {
            break;  /* EOF or error */
        }
        
        /* Expand buffer if needed */
        if (total_size >= buffer_size - 1) {
            buffer_size *= 2;
            char* new_buffer = realloc(buffer, buffer_size);
            if (!new_buffer) {
                fprintf(stderr, "❌ Failed to expand input buffer\n");
                free(buffer);
                return NULL;
            }
            buffer = new_buffer;
        }
    }
    
    buffer[total_size] = '\0';
    
    /* Shrink buffer to actual size */
    if (total_size > 0) {
        char* final_buffer = realloc(buffer, total_size + 1);
        if (final_buffer) {
            buffer = final_buffer;
        }
    }
    
    printf("📥 Read %zu bytes from stdin\n", total_size);
    return buffer;
}

/* Read file content into string */
char* read_file_content(const char* filename) {
    if (!filename) {
        return NULL;
    }
    
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "❌ Cannot open file '%s': %s\n", filename, strerror(errno));
        return NULL;
    }
    
    /* Get file size */
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    if (file_size <= 0) {
        fclose(file);
        fprintf(stderr, "⚠️  File '%s' is empty or cannot determine size\n", filename);
        return NULL;
    }
    
    /* Allocate buffer */
    char* content = malloc(file_size + 1);
    if (!content) {
        fclose(file);
        fprintf(stderr, "❌ Failed to allocate memory for file '%s'\n", filename);
        return NULL;
    }
    
    /* Read file */
    size_t bytes_read = fread(content, 1, file_size, file);
    content[bytes_read] = '\0';
    fclose(file);
    
    printf("📄 Read %zu bytes from file '%s'\n", bytes_read, filename);
    return content;
}

/* Process file input through Galileo model */
int process_file_input(GalileoModel* model, const char* filename) {
    if (!model || !filename) {
        return -1;
    }
    
    char* content = read_file_content(filename);
    if (!content) {
        return -1;
    }
    
    /* Tokenize content */
    int token_count;
    char** tokens = tokenize_input(content, &token_count);
    
    if (tokens && token_count > 0) {
        /* Convert to format expected by processing functions */
        char token_array[token_count][MAX_TOKEN_LEN];
        for (int i = 0; i < token_count; i++) {
            strncpy(token_array[i], tokens[i], MAX_TOKEN_LEN - 1);
            token_array[i][MAX_TOKEN_LEN - 1] = '\0';
        }
        
        /* Process through Galileo */
        galileo_process_sequence(model, token_array, token_count);
        
        free_tokens(tokens, token_count);
    }
    
    free(content);
    return 0;
}

/* =============================================================================
 * OUTPUT FORMATTING AND DISPLAY
 * =============================================================================
 */

/* Print comprehensive model summary */
void print_model_summary(GalileoModel* model, FILE* output) {
    if (!model || !output) return;
    
    fprintf(output, "\n📊 === Galileo v42 Model Summary ===\n");
    
    /* Basic statistics */
    int active_nodes = 0, active_edges = 0, active_memories = 0;
    for (int i = 0; i < model->num_nodes; i++) {
        if (model->nodes[i].active) active_nodes++;
    }
    for (int i = 0; i < model->num_edges; i++) {
        if (model->edges[i].active) active_edges++;
    }
    for (int i = 0; i < model->num_memory_slots; i++) {
        if (model->memory_slots[i].active) active_memories++;
    }
    
    fprintf(output, "\n🧠 Graph Structure:\n");
    fprintf(output, "  Nodes: %d/%d active (%.1f%% utilization)\n", 
            active_nodes, MAX_TOKENS, (100.0f * active_nodes) / MAX_TOKENS);
    fprintf(output, "  Edges: %d/%d active (%.1f%% utilization)\n", 
            active_edges, MAX_EDGES, (100.0f * active_edges) / MAX_EDGES);
    fprintf(output, "  Average degree: %.2f\n", model->avg_node_degree);
    fprintf(output, "  Global node: %d\n", model->global_node_idx);
    fprintf(output, "  Attention hubs: %d\n", model->num_attention_hubs);
    
    fprintf(output, "\n💾 Memory System:\n");
    fprintf(output, "  Memory slots: %d/%d active (%.1f%% utilization)\n", 
            active_memories, MAX_MEMORY_SLOTS, (100.0f * active_memories) / MAX_MEMORY_SLOTS);
    fprintf(output, "  Total compressions: %d\n", model->total_compressions);
    
    fprintf(output, "\n🧠 Symbolic Knowledge:\n");
    fprintf(output, "  Facts: %d/%d (%.1f%% utilization)\n", 
            model->num_facts, MAX_FACTS, (100.0f * model->num_facts) / MAX_FACTS);
    fprintf(output, "  Conflicts resolved: %d\n", model->num_resolved_conflicts);
    fprintf(output, "  Symbolic calls: %d\n", model->total_symbolic_calls);
    
    fprintf(output, "\n⚙️  Configuration:\n");
    fprintf(output, "  Similarity threshold: %.3f\n", model->similarity_threshold);
    fprintf(output, "  Attention threshold: %.3f\n", model->attention_threshold);
    fprintf(output, "  Compression threshold: %.3f\n", model->compression_threshold);
    fprintf(output, "  Max iterations: %d\n", model->max_iterations);
    fprintf(output, "  Current iteration: %d\n", model->current_iteration);
    
    fprintf(output, "\n📈 Performance Metrics:\n");
    fprintf(output, "  Total edges added: %d\n", model->total_edges_added);
    fprintf(output, "  Vocabulary size: %d\n", model->vocab_size);
    fprintf(output, "  Importance decay: %.3f\n", model->importance_decay);
    
    fprintf(output, "\n");
}

/* Print all learned facts in a beautiful format */
void print_facts(GalileoModel* model, FILE* output) {
    if (!model || !output) return;
    
    fprintf(output, "\n💡 === Learned Facts ===\n");
    
    if (model->num_facts == 0) {
        fprintf(output, "  (No facts learned yet)\n\n");
        return;
    }
    
    fprintf(output, "\nFacts (%d total):\n", model->num_facts);
    
    /* Sort facts by confidence (simple bubble sort) */
    typedef struct {
        int index;
        float confidence;
        int derived;
    } FactInfo;
    
    FactInfo fact_infos[MAX_FACTS];
    for (int i = 0; i < model->num_facts; i++) {
        fact_infos[i].index = i;
        fact_infos[i].confidence = model->facts[i].confidence;
        fact_infos[i].derived = model->facts[i].derived;
    }
    
    /* Sort by confidence (descending) */
    for (int i = 0; i < model->num_facts - 1; i++) {
        for (int j = 0; j < model->num_facts - i - 1; j++) {
            if (fact_infos[j].confidence < fact_infos[j + 1].confidence) {
                FactInfo temp = fact_infos[j];
                fact_infos[j] = fact_infos[j + 1];
                fact_infos[j + 1] = temp;
            }
        }
    }
    
    /* Print sorted facts */
    for (int i = 0; i < model->num_facts; i++) {
        SymbolicFact* fact = &model->facts[fact_infos[i].index];
        
        /* Create visual confidence bar using ASCII characters */
        char confidence_bar[11];
        int bar_length = (int)(fact->confidence * 10);
        for (int j = 0; j < 10; j++) {
            confidence_bar[j] = (j < bar_length) ? '#' : '-';
        }
        confidence_bar[10] = '\0';
        
        const char* type_marker = fact->derived ? "[DERIVED]" : "[DIRECT] ";
        
        fprintf(output, "  %s %s %s %s (%.2f) [%s]\n",
                type_marker, fact->subject, fact->relation, fact->object, 
                fact->confidence, confidence_bar);
        
        if (fact->support_count > 0) {
            fprintf(output, "    └─ Supported by %d nodes\n", fact->support_count);
        }
    }
    
    fprintf(output, "\nLegend: [DIRECT]  = Direct facts  [DERIVED] = Derived facts\n\n");
}

/* Print graph statistics in a detailed format */
void print_graph_stats(GalileoModel* model, FILE* output) {
    if (!model || !output) return;
    
    fprintf(output, "\n📈 === Graph Statistics ===\n");
    
    /* Node analysis */
    int node_types[5] = {0}; /* token, global, summary, query, other */
    float total_importance = 0.0f;
    int active_nodes = 0;
    
    for (int i = 0; i < model->num_nodes; i++) {
        if (!model->nodes[i].active) continue;
        
        active_nodes++;
        total_importance += model->nodes[i].importance_score;
        
        /* Classify node type */
        if (i == model->global_node_idx) {
            node_types[1]++; /* global */
        } else if (strstr(model->nodes[i].token_text, "[SUMMARY")) {
            node_types[2]++; /* summary */
        } else if (strstr(model->nodes[i].token_text, "[QUERY")) {
            node_types[3]++; /* query */
        } else if (model->nodes[i].token_text[0] == '[') {
            node_types[4]++; /* other special */
        } else {
            node_types[0]++; /* regular token */
        }
    }
    
    fprintf(output, "\n🎯 Node Analysis:\n");
    fprintf(output, "  Token nodes: %d\n", node_types[0]);
    fprintf(output, "  Global nodes: %d\n", node_types[1]);
    fprintf(output, "  Summary nodes: %d\n", node_types[2]);
    fprintf(output, "  Query nodes: %d\n", node_types[3]);
    fprintf(output, "  Other nodes: %d\n", node_types[4]);
    fprintf(output, "  Average importance: %.3f\n", 
            active_nodes > 0 ? total_importance / active_nodes : 0.0f);
    
    /* Edge analysis */
    int edge_types[6] = {0}; /* sequence, similarity, attention, causal, semantic, other */
    float total_weight = 0.0f;
    int active_edges = 0;
    
    for (int i = 0; i < model->num_edges; i++) {
        if (!model->edges[i].active) continue;
        
        active_edges++;
        total_weight += model->edges[i].weight;
        
        /* Classify edge type */
        switch (model->edges[i].type) {
            case EDGE_SEQUENCE: edge_types[0]++; break;
            case EDGE_SIMILARITY: edge_types[1]++; break;
            case EDGE_ATTENTION: edge_types[2]++; break;
            case EDGE_CAUSAL: edge_types[3]++; break;
            case EDGE_SEMANTIC: edge_types[4]++; break;
            default: edge_types[5]++; break;
        }
    }
    
    fprintf(output, "\n🔗 Edge Analysis:\n");
    fprintf(output, "  Sequence edges: %d\n", edge_types[0]);
    fprintf(output, "  Similarity edges: %d\n", edge_types[1]);
    fprintf(output, "  Attention edges: %d\n", edge_types[2]);
    fprintf(output, "  Causal edges: %d\n", edge_types[3]);
    fprintf(output, "  Semantic edges: %d\n", edge_types[4]);
    fprintf(output, "  Other edges: %d\n", edge_types[5]);
    fprintf(output, "  Average weight: %.3f\n", 
            active_edges > 0 ? total_weight / active_edges : 0.0f);
    
    /* Connectivity analysis */
    fprintf(output, "\n🌐 Connectivity:\n");
    fprintf(output, "  Average degree: %.2f\n", model->avg_node_degree);
    fprintf(output, "  Sparsity: %.6f\n", 
            (float)active_edges / (active_nodes * active_nodes));
    fprintf(output, "  Edge density: %.1f%%\n", 
            (100.0f * active_edges) / MAX_EDGES);
    
    /* Memory analysis */
    int active_memories = 0;
    float avg_memory_importance = 0.0f;
    
    for (int i = 0; i < model->num_memory_slots; i++) {
        if (model->memory_slots[i].active) {
            active_memories++;
            avg_memory_importance += model->memory_slots[i].importance;
        }
    }
    
    fprintf(output, "\n💾 Memory Analysis:\n");
    fprintf(output, "  Active memory slots: %d/%d\n", active_memories, MAX_MEMORY_SLOTS);
    fprintf(output, "  Average memory importance: %.3f\n", 
            active_memories > 0 ? avg_memory_importance / active_memories : 0.0f);
    fprintf(output, "  Memory utilization: %.1f%%\n", 
            (100.0f * active_memories) / MAX_MEMORY_SLOTS);
    
    fprintf(output, "\n");
}
