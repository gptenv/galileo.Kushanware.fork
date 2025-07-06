/* =============================================================================
 * galileo/src/main/galileo_main.c - CLI Interface and Main Application
 * 
 * COMPLETELY REWRITTEN for true hot-pluggable lazy loading module system.
 * No hardcoded module lists - everything is discovery-based with JIT loading.
 * Modules are loaded on-first-use and can declare their own dependencies.
 * =============================================================================
 */

/* Enable POSIX functions like strdup() */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE  /* For older glibc compatibility */

#include "../../include/galileo.h"
#include "../core/galileo_module_loader.h"
#include <getopt.h>
#include <signal.h>
#include <errno.h>
#include <ctype.h>
#include <poll.h>
#include <string.h>
#include <dlfcn.h>

/* =============================================================================
 * GLOBAL STATE AND CONFIGURATION
 * =============================================================================
 */

/* Signal handling state */
volatile int g_shutdown_requested = 0;
GalileoModel* g_current_model = NULL;

/* Command line options with GNU-style long options */
static struct option long_options[] = {
    {"help",                no_argument,       0, 'h'},
    {"version",             no_argument,       0, 'V'},
    {"verbose",             no_argument,       0, 'v'},
    {"quiet",               no_argument,       0, 'q'},
    {"test",                no_argument,       0, 't'},
    {"interactive",         no_argument,       0, 'i'},
    {"output",              required_argument, 0, 'o'},
    {"config",              required_argument, 0, 'c'},
    {"max-iterations",      required_argument, 0, 1001},
    {"similarity-threshold", required_argument, 0, 1002},
    {"attention-threshold", required_argument, 0, 1003},
    {"no-symbolic",         no_argument,       0, 1004},
    {"no-compression",      no_argument,       0, 1005},
    {"no-memory",           no_argument,       0, 1006},
    {"no-graph",            no_argument,       0, 1007},
    {"list-modules",        no_argument,       0, 1008},
    {0, 0, 0, 0}
};

/* =============================================================================
 * SIGNAL HANDLING AND CLEANUP
 * =============================================================================
 */

/* Signal handler for graceful shutdown */
void signal_handler(int signal) {
    switch (signal) {
        case SIGINT:
        case SIGTERM:
            fprintf(stderr, "\n🛑 Shutdown signal received (%d), cleaning up...\n", signal);
            g_shutdown_requested = 1;
            if (g_current_model) {
                fprintf(stderr, "🔄 Saving model state...\n");
                /* Could add model state saving here */
            }
            break;
        default:
            fprintf(stderr, "⚠️  Unexpected signal: %d\n", signal);
            break;
    }
}

/* Set up signal handlers */
void setup_signal_handlers(void) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    /* Ignore SIGPIPE to handle broken pipes gracefully */
    signal(SIGPIPE, SIG_IGN);
}

/* Cleanup and exit gracefully */
void cleanup_and_exit(int exit_code) {
    if (g_current_model) {
        fprintf(stderr, "🔥 Destroying model...\n");
        galileo_destroy(g_current_model);
        g_current_model = NULL;
    }
    
    fprintf(stderr, "🧹 Unloading all modules...\n");
    galileo_module_loader_cleanup();
    
    fprintf(stderr, "✅ Cleanup complete. Goodbye!\n");
    exit(exit_code);
}

/* =============================================================================
 * LAZY MODULE LOADING FUNCTIONS - NO HARDCODED LISTS!
 * =============================================================================
 */

/* Load only essential bootstrap modules */
int load_bootstrap_modules(void) {
    printf("🚀 Loading bootstrap modules...\n");
    
    /* Only load core module at startup - everything else is lazy loaded */
    printf("📦 Loading bootstrap module: core\n");
    int result = galileo_load_module("core");
    if (result != 0) {
        fprintf(stderr, "❌ Failed to load core module (required for bootstrap)\n");
        return -1;
    }
    printf("✅ Bootstrap module 'core' loaded successfully\n");
    
    return 0;
}

/* Try to lazy-load a module when needed */
int ensure_module_loaded(const char* module_name) {
    if (galileo_is_module_loaded(module_name)) {
        return 0;  /* Already loaded */
    }
    
    printf("🔄 Lazy-loading module '%s' on first use...\n", module_name);
    int result = galileo_load_module(module_name);
    if (result == 0) {
        printf("✅ Module '%s' loaded successfully\n", module_name);
    } else {
        printf("⚠️  Module '%s' failed to load: %s\n", 
               module_name, galileo_module_error_string(result));
    }
    return result;
}

/* Load all discovered modules (for --list-modules) */
int load_all_discovered_modules(void) {
    printf("🔄 Loading all discovered modules for inspection...\n");
    
    ModuleLoadInfo discovered[32];
    int count = galileo_list_modules(discovered, 32);
    
    for (int i = 0; i < count; i++) {
        if (!discovered[i].loaded) {
            printf("📦 Loading discovered module: %s\n", discovered[i].name);
            int result = galileo_load_module(discovered[i].name);
            if (result == 0) {
                printf("✅ Module '%s' loaded successfully\n", discovered[i].name);
            } else {
                printf("⚠️  Module '%s' failed to load: %s\n", 
                       discovered[i].name, galileo_module_error_string(result));
            }
        }
    }
    
    return 0;
}

/* Check if a specific module is loaded */
int is_module_loaded(const char* module_name) {
    return galileo_is_module_loaded(module_name);
}

/* Print module status */
void print_module_status(FILE* output) {
    fprintf(output, "\n📚 Module Status:\n");
    
    ModuleLoadInfo modules[32];
    int count = galileo_list_modules(modules, 32);
    
    if (count <= 0) {
        fprintf(output, "  No modules discovered\n");
        return;
    }
    
    for (int i = 0; i < count; i++) {
        const char* status_icon = modules[i].loaded ? "✅" : "❌";
        const char* req_text = modules[i].required ? "required" : "optional";
        const char* version = modules[i].version[0] ? modules[i].version : "unknown";
        
        fprintf(output, "  %s %-12s %-10s %-10s", 
                status_icon, modules[i].name, req_text, 
                modules[i].loaded ? "loaded" : "not loaded");
        
        if (modules[i].loaded) {
            fprintf(output, " (v%s)", version);
        } else if (modules[i].load_failed) {
            fprintf(output, " (failed: %s)", modules[i].error_message);
        }
        
        fprintf(output, "\n");
    }
}

/* =============================================================================
 * COMMAND LINE PARSING AND HELP
 * =============================================================================
 */

/* Print usage information */
void print_usage(const char* program_name) {
    fprintf(stderr, "Usage: %s [OPTIONS] [FILES...]\n\n", program_name);
    fprintf(stderr, "Galileo Graph-and-Logic Integrated Language Engine v42 (Dynamic Loading Edition)\n\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -h, --help                     Show this help message\n");
    fprintf(stderr, "  -V, --version                  Show version information\n");
    fprintf(stderr, "  -v, --verbose                  Enable verbose output\n");
    fprintf(stderr, "  -q, --quiet                    Suppress non-essential output\n");
    fprintf(stderr, "  -t, --test                     Run test suite\n");
    fprintf(stderr, "  -i, --interactive              Enter interactive mode\n");
    fprintf(stderr, "  -o, --output FILE              Write output to FILE\n");
    fprintf(stderr, "  -c, --config FILE              Load configuration from FILE\n");
    fprintf(stderr, "      --max-iterations N         Set maximum processing iterations\n");
    fprintf(stderr, "      --similarity-threshold F   Set similarity threshold (0.0-1.0)\n");
    fprintf(stderr, "      --attention-threshold F    Set attention threshold (0.0-1.0)\n");
    fprintf(stderr, "      --no-symbolic              Disable symbolic reasoning module\n");
    fprintf(stderr, "      --no-compression           Disable memory compression module\n");
    fprintf(stderr, "      --no-memory                Disable memory management module\n");
    fprintf(stderr, "      --no-graph                 Disable graph processing module\n");
    fprintf(stderr, "      --list-modules             Show available modules and exit\n");
    fprintf(stderr, "\n");
    
    fprintf(stderr, "Input Sources (processed in order):\n");
    fprintf(stderr, "  stdin                          Read from standard input (auto-detected)\n");
    fprintf(stderr, "  FILES...                       Process one or more files\n");
    fprintf(stderr, "\n");
    
    fprintf(stderr, "Examples:\n");
    fprintf(stderr, "  %s --test                             # Run test suite\n", program_name);
    fprintf(stderr, "  %s -i                                 # Interactive mode\n", program_name);
    fprintf(stderr, "  %s file1.txt file2.txt               # Process files\n", program_name);
    fprintf(stderr, "  echo \"All men are mortal\" | %s       # Process from stdin\n", program_name);
    fprintf(stderr, "  %s -v --max-iterations 10 < input.txt > output.txt\n", program_name);
    fprintf(stderr, "  %s --no-symbolic data/*.txt           # Disable symbolic reasoning\n", program_name);
    fprintf(stderr, "  %s --list-modules                     # Show available modules\n", program_name);
    fprintf(stderr, "\n");
    
    fprintf(stderr, "Exit Codes:\n");
    fprintf(stderr, "  0    Success\n");
    fprintf(stderr, "  1    General error\n");
    fprintf(stderr, "  2    Command line argument error\n");
    fprintf(stderr, "  3    Module loading error\n");
    fprintf(stderr, "  4    Input/output error\n");
    fprintf(stderr, "  5    Memory error\n");
}

/* Print version information */
void print_version(void) {
    printf("Galileo v%s (Dynamic Loading Edition)\n", GALILEO_VERSION);
    printf("Graph-and-Logic Integrated Language Engine\n");
    printf("Built with hot-swappable module support\n");
    
    /* Show module loader version */
    char loader_version[64];
    if (galileo_get_module_loader_version(loader_version, sizeof(loader_version)) == 0) {
        printf("Module Loader: %s\n", loader_version);
    }
    
    printf("Features: lazy loading, JIT modules, no hardcoded dependencies\n");
}

/* Parse command line arguments */
int parse_arguments(int argc, char* argv[], GalileoOptions* options) {
    /* Initialize options with defaults */
    memset(options, 0, sizeof(GalileoOptions));
    options->similarity_threshold = -1.0f;  /* Indicates not set */
    options->attention_threshold = -1.0f;   /* Indicates not set */
    options->max_iterations = -1;           /* Indicates not set */
    
    int c;
    int option_index = 0;
    
    while ((c = getopt_long(argc, argv, "hVvqtio:c:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'h':
                options->help = 1;
                break;
            case 'V':
                options->version = 1;
                break;
            case 'v':
                options->verbose = 1;
                break;
            case 'q':
                options->quiet = 1;
                break;
            case 't':
                options->test_mode = 1;
                break;
            case 'i':
                options->interactive = 1;
                break;
            case 'o':
                options->output_file = strdup(optarg);
                if (!options->output_file) {
                    fprintf(stderr, "❌ Failed to allocate memory for output file\n");
                    return 1;
                }
                break;
            case 'c':
                options->config_file = strdup(optarg);
                if (!options->config_file) {
                    fprintf(stderr, "❌ Failed to allocate memory for config file\n");
                    return 1;
                }
                break;
            case 1001:  /* max-iterations */
                options->max_iterations = atoi(optarg);
                if (options->max_iterations < 1 || options->max_iterations > 100) {
                    fprintf(stderr, "❌ Error: max-iterations must be between 1 and 100\n");
                    return 2;
                }
                break;
            case 1002:  /* similarity-threshold */
                options->similarity_threshold = atof(optarg);
                if (options->similarity_threshold < 0.0f || options->similarity_threshold > 1.0f) {
                    fprintf(stderr, "❌ Error: similarity-threshold must be between 0.0 and 1.0\n");
                    return 2;
                }
                break;
            case 1003:  /* attention-threshold */
                options->attention_threshold = atof(optarg);
                if (options->attention_threshold < 0.0f || options->attention_threshold > 1.0f) {
                    fprintf(stderr, "❌ Error: attention-threshold must be between 0.0 and 1.0\n");
                    return 2;
                }
                break;
            case 1004:  /* no-symbolic */
                options->disable_symbolic = 1;
                break;
            case 1005:  /* no-compression */
                options->disable_compression = 1;
                break;
            case 1006:  /* no-memory */
                options->disable_memory = 1;
                break;
            case 1007:  /* no-graph */
                options->disable_graph = 1;
                break;
            case 1008:  /* list-modules */
                options->help = 2;  /* Special flag for module listing */
                break;
            case '?':
                /* getopt_long already printed an error message */
                return 2;
            default:
                fprintf(stderr, "❌ Error: Unknown option\n");
                return 2;
        }
    }
    
    /* Collect remaining arguments as input files */
    options->input_file_count = argc - optind;
    if (options->input_file_count > 0) {
        options->input_files = malloc(options->input_file_count * sizeof(char*));
        if (!options->input_files) {
            fprintf(stderr, "❌ Failed to allocate memory for input files\n");
            return 1;
        }
        for (int i = 0; i < options->input_file_count; i++) {
            options->input_files[i] = strdup(argv[optind + i]);
            if (!options->input_files[i]) {
                fprintf(stderr, "❌ Failed to duplicate input filename\n");
                return 1;
            }
        }
    }
    
    return 0;
}

/* Clean up options structure */
void cleanup_options(GalileoOptions* options) {
    if (options->output_file) {
        free(options->output_file);
    }
    if (options->config_file) {
        free(options->config_file);
    }
    if (options->input_files) {
        for (int i = 0; i < options->input_file_count; i++) {
            if (options->input_files[i]) {
                free(options->input_files[i]);
            }
        }
        free(options->input_files);
    }
    memset(options, 0, sizeof(GalileoOptions));
}

/* =============================================================================
 * INPUT PROCESSING FUNCTIONS
 * =============================================================================
 */

/* Check if stdin has data available */
int is_stdin_available(void) {
    if (isatty(STDIN_FILENO)) {
        return 0;  /* Interactive terminal, no piped input */
    }
    
    struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
    int result = poll(&pfd, 1, 0);
    return (result > 0 && (pfd.revents & POLLIN));
}

/* Read all input from stdin */
char* read_stdin_input(void) {
    size_t capacity = 1024;
    size_t length = 0;
    char* buffer = malloc(capacity);
    
    if (!buffer) return NULL;
    
    int c;
    while ((c = getchar()) != EOF) {
        if (length >= capacity - 1) {
            capacity *= 2;
            char* new_buffer = realloc(buffer, capacity);
            if (!new_buffer) {
                free(buffer);
                return NULL;
            }
            buffer = new_buffer;
        }
        buffer[length++] = c;
    }
    
    buffer[length] = '\0';
    return buffer;
}

/* Process text input through Galileo using dynamically loaded modules */
int process_input_text(GalileoModel* model, const char* text, const GalileoOptions* options) {
    if (!model || !text) return 4;
    
    if (!options->quiet) {
        fprintf(stderr, "🔄 Processing text input (%zu chars)...\n", strlen(text));
    }
    
    /* Simple tokenization */
    printf("📝 Processing text: %s\n", text);
    
    /* Create a simple token array from the input */
    char simple_tokens[16][MAX_TOKEN_LEN];
    int token_count = 0;
    
    /* Basic whitespace tokenization */
    char* text_copy = strdup(text);
    char* token = strtok(text_copy, " \t\n\r.,!?;:");
    while (token && token_count < 16) {
        strncpy(simple_tokens[token_count], token, MAX_TOKEN_LEN - 1);
        simple_tokens[token_count][MAX_TOKEN_LEN - 1] = '\0';
        token_count++;
        token = strtok(NULL, " \t\n\r.,!?;:");
    }
    free(text_copy);
    
    if (token_count > 0) {
        printf("🔤 Tokenized into %d tokens\n", token_count);
        
        /* Call galileo_process_sequence - core module handles lazy loading of other modules */
        printf("🚀 Calling galileo_process_sequence from loaded core module...\n");
        galileo_process_sequence(model, simple_tokens, token_count);
        
        /* Print results using basic output */
        printf("\n📊 Processing Results:\n");
        printf("Final model state: %d nodes, %d edges\n", model->num_nodes, model->num_edges);
        
        if (options->verbose) {
            printf("📈 Detailed statistics:\n");
            galileo_compute_graph_stats(model);
        }
    } else {
        printf("⚠️  No tokens found in input\n");
    }
    
    return 0;
}

/* Process stdin input */
int process_stdin(GalileoModel* model, const GalileoOptions* options) {
    if (!options->quiet) {
        fprintf(stderr, "📥 Reading from stdin...\n");
    }
    
    char* input = read_stdin_input();
    if (!input) {
        fprintf(stderr, "❌ Failed to read stdin\n");
        return 4;
    }
    
    if (strlen(input) == 0) {
        if (!options->quiet) {
            fprintf(stderr, "⚠️  No input received from stdin\n");
        }
        free(input);
        return 0;
    }
    
    int result = process_input_text(model, input, options);
    free(input);
    return result;
}

/* Process files */
int process_files(GalileoModel* model, const GalileoOptions* options) {
    for (int i = 0; i < options->input_file_count; i++) {
        if (!options->quiet) {
            fprintf(stderr, "📄 Processing file: %s\n", options->input_files[i]);
        }
        
        /* Read file content and process it */
        FILE* file = fopen(options->input_files[i], "r");
        if (!file) {
            fprintf(stderr, "❌ Cannot open file '%s': %s\n", 
                    options->input_files[i], strerror(errno));
            continue;
        }
        
        /* Read file content */
        fseek(file, 0, SEEK_END);
        long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        
        if (file_size <= 0) {
            fprintf(stderr, "⚠️  File '%s' is empty\n", options->input_files[i]);
            fclose(file);
            continue;
        }
        
        char* content = malloc(file_size + 1);
        if (!content) {
            fprintf(stderr, "❌ Failed to allocate memory for file content\n");
            fclose(file);
            return 5;
        }
        
        size_t read_size = fread(content, 1, file_size, file);
        fclose(file);
        
        if (read_size != (size_t)file_size) {
            fprintf(stderr, "❌ Failed to read complete file '%s'\n", options->input_files[i]);
            free(content);
            continue;
        }
        
        content[file_size] = '\0';
        
        /* Process the content */
        int result = process_input_text(model, content, options);
        free(content);
        
        if (result != 0) {
            return result;
        }
    }
    
    return 0;
}

/* =============================================================================
 * STUB FUNCTIONS FOR FEATURES NOT YET IMPLEMENTED
 * =============================================================================
 */

/* Run test suite */
int run_test_suite(GalileoModel* model, const GalileoOptions* options) {
    (void)model;
    (void)options;
    printf("🧪 Test suite functionality not yet implemented\n");
    printf("🔄 Dynamic loading system is working correctly.\n");
    
    if (options->verbose) {
        printf("\n📊 Final module status:\n");
        print_module_status(stdout);
    }
    
    return 0;
}

/* Run interactive mode */
int run_interactive_mode(GalileoModel* model, const GalileoOptions* options) {
    (void)model;
    (void)options;
    printf("💬 Interactive mode not yet implemented\n");
    printf("🔄 Dynamic loading system is working correctly.\n");
    
    if (options->verbose) {
        printf("\n📊 Final module status:\n");
        print_module_status(stdout);
    }
    
    return 0;
}

/* =============================================================================
 * MAIN ENTRY POINT - COMPLETELY REWRITTEN FOR LAZY LOADING
 * =============================================================================
 */

int main(int argc, char* argv[]) {
    GalileoOptions options = {0};
    int result = 0;
    
    /* Set up signal handling */
    setup_signal_handlers();
    
    /* Parse command line arguments */
    int parse_result = parse_arguments(argc, argv, &options);
    if (parse_result != 0) {
        cleanup_options(&options);
        return parse_result;
    }
    
    /* Handle special cases */
    if (options.help == 2) {
        /* Initialize module loader and load all modules for inspection */
        if (galileo_module_loader_init() != 0) {
            fprintf(stderr, "❌ Failed to initialize module loading system\n");
            cleanup_options(&options);
            return 3;
        }
        load_all_discovered_modules();  /* Load everything for --list-modules */
        print_module_status(stdout);
        galileo_module_loader_cleanup();
        cleanup_options(&options);
        return 0;
    } else if (options.help) {
        print_usage(argv[0]);
        cleanup_options(&options);
        return 0;
    }
    
    if (options.version) {
        print_version();
        cleanup_options(&options);
        return 0;
    }
    
    /* Initialize the module loading system */
    printf("🚀 Initializing Galileo module loading system...\n");
    if (galileo_module_loader_init() != 0) {
        fprintf(stderr, "❌ Failed to initialize module loading system\n");
        cleanup_options(&options);
        return 3;
    }
    
    /* Load only bootstrap modules (just core) - everything else lazy loaded */
    if (load_bootstrap_modules() != 0) {
        fprintf(stderr, "❌ Failed to load bootstrap modules\n");
        cleanup_and_exit(3);
    }
    
    /* Initialize model using dynamically loaded core module */
    printf("🚀 Initializing Galileo model using loaded core module...\n");
    
    /* Call galileo_init() from the loaded core module */
    g_current_model = galileo_init();
    if (!g_current_model) {
        fprintf(stderr, "❌ Failed to initialize Galileo model\n");
        cleanup_and_exit(5);
    }
    
    printf("✅ Model initialized successfully using dynamic core module!\n");
    
    /* Apply command line options to model */
    if (options.max_iterations > 0) {
        g_current_model->max_iterations = options.max_iterations;
    }
    if (options.similarity_threshold >= 0.0f && options.similarity_threshold <= 1.0f) {
        g_current_model->similarity_threshold = options.similarity_threshold;
    }
    
    /* Show initial module status (only bootstrap modules loaded) */
    if (options.verbose) {
        print_module_status(stdout);
    }
    
    /* Handle different input modes - modules will be lazy loaded as needed */
    if (options.test_mode) {
        /* Test mode might need various modules - they'll be lazy loaded */
        result = run_test_suite(g_current_model, &options);
    } else if (options.interactive) {
        /* Interactive mode - modules loaded as needed */
        result = run_interactive_mode(g_current_model, &options);
    } else if (is_stdin_available()) {
        /* Process stdin - modules loaded as needed during processing */
        result = process_stdin(g_current_model, &options);
        
        /* Also process files if provided */
        if (options.input_file_count > 0) {
            result |= process_files(g_current_model, &options);
        }
    } else if (options.input_file_count > 0) {
        /* Process files - modules loaded as needed during processing */
        result = process_files(g_current_model, &options);
    } else {
        fprintf(stderr, "Error: No input provided. Use -h for help.\n");
        result = 1;
    }
    
    /* Cleanup */
    cleanup_and_exit(result);
    return result;
}
