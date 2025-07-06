/* =============================================================================
 * galileo/src/heuristic/galileo_heuristic_compiler.c - Heuristic Compiler Module
 * 
 * Hot-loadable shared library implementing GA-derived heuristic rule caching,
 * SQLite persistence, and self-improving fact extraction. This module learns
 * from genetic algorithm discoveries and compiles them into fast lookup rules
 * that persist across restarts and improve over time.
 * 
 * Uses hybrid approach: lightning-fast cache lookups for known patterns,
 * fallback to GA discovery for novel patterns, with 19-year knowledge preservation.
 * =============================================================================
 */

/* Feature test macros for maximum portability */
#if defined(__linux__) || defined(__GLIBC__)
    #define _GNU_SOURCE
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
    #define _POSIX_C_SOURCE 200112L
    #define _DEFAULT_SOURCE
#elif defined(__sun) || defined(__SVR4)
    #define _POSIX_C_SOURCE 200112L
    #define __EXTENSIONS__
#else
    #define _POSIX_C_SOURCE 200112L
#endif

#include "galileo_heuristic_compiler.h"
#include "../core/galileo_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sqlite3.h>
#include <unistd.h>

/* Thread support - portable across UNIX systems */
#if defined(_POSIX_THREADS) || defined(__APPLE__) || defined(__linux__)
    #include <pthread.h>
    #define GALILEO_HAS_PTHREAD 1
#else
    #define GALILEO_HAS_PTHREAD 0
#endif

/* =============================================================================
 * MODULE METADATA AND INITIALIZATION
 * =============================================================================
 */

static int heuristic_module_initialized = 0;
static HeuristicCompiler* g_compiler = NULL;

#if GALILEO_HAS_PTHREAD
static pthread_mutex_t compiler_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif

/* Module initialization */
static int heuristic_module_init(void) {
    if (heuristic_module_initialized) {
        return 0;  /* Already initialized */
    }
    
    fprintf(stderr, "🧬 Heuristic compiler module v42.1 initializing...\n");
    
    /* Initialize the global compiler */
    g_compiler = create_heuristic_compiler("./galileo_knowledge.db");
    if (!g_compiler) {
        fprintf(stderr, "❌ Failed to initialize heuristic compiler\n");
        return -1;
    }
    
    heuristic_module_initialized = 1;
    fprintf(stderr, "✅ Heuristic compiler ready! GA-derived rule cache online.\n");
    return 0;
}

/* Module cleanup */
static void heuristic_module_cleanup(void) {
    if (!heuristic_module_initialized) {
        return;
    }
    
    fprintf(stderr, "🧬 Heuristic compiler module shutting down...\n");
    
    if (g_compiler) {
        destroy_heuristic_compiler(g_compiler);
        g_compiler = NULL;
    }
    
    heuristic_module_initialized = 0;
}

/* Module info structure for dynamic loading */
HeuristicModuleInfo heuristic_module_info = {
    .name = "heuristic",
    .version = "42.1",
    .init_func = heuristic_module_init,
    .cleanup_func = heuristic_module_cleanup
};

/* =============================================================================
 * GENETIC ALGORITHM IMPLEMENTATION
 * =============================================================================
 */

/* Simple genetic algorithm for fact role discovery */
typedef struct {
    int subject_idx;    /* Which token index is subject */
    int relation_idx;   /* Which token index is relation */  
    int object_idx;     /* Which token index is object */
    float fitness;      /* Fitness score (inverse of energy) */
} FactChromosome;

typedef struct {
    FactChromosome population[GA_POPULATION_SIZE];
    int generation;
    int converged;
    FactChromosome best_solution;
} GeneticAlgorithm;

/* Calculate fitness for a chromosome (lower energy = higher fitness) */
static float calculate_fitness(GalileoModel* model, FactChromosome* chromo, 
                              char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    if (chromo->subject_idx == chromo->relation_idx || 
        chromo->subject_idx == chromo->object_idx || 
        chromo->relation_idx == chromo->object_idx ||
        chromo->subject_idx >= num_tokens ||
        chromo->relation_idx >= num_tokens ||
        chromo->object_idx >= num_tokens) {
        return 0.0f;  /* Invalid assignment */
    }
    
    float energy = 0.0f;
    
    /* Energy term 1: Semantic coherence (subject and object should be related) */
    /* For now, use position preference and length heuristics */
    float position_energy = fabsf(chromo->subject_idx - chromo->object_idx) * 0.1f;
    
    /* Energy term 2: Relation quality (middle tokens often better relations) */
    float relation_centrality = fabsf(chromo->relation_idx - (num_tokens / 2.0f));
    
    /* Energy term 3: Common linguistic patterns (subject-relation-object order) */
    float order_energy = 0.0f;
    if (chromo->subject_idx < chromo->relation_idx && chromo->relation_idx < chromo->object_idx) {
        order_energy = -0.5f;  /* Bonus for SVO order */
    }
    
    energy = position_energy + relation_centrality * 0.2f + order_energy;
    
    /* Fitness is inverse of energy */
    return 1.0f / (1.0f + energy);
}

/* Initialize random population */
static void init_population(GeneticAlgorithm* ga, int num_tokens) {
    for (int i = 0; i < GA_POPULATION_SIZE; i++) {
        FactChromosome* chromo = &ga->population[i];
        
        /* Random permutation of token indices */
        chromo->subject_idx = rand() % num_tokens;
        chromo->relation_idx = rand() % num_tokens;
        chromo->object_idx = rand() % num_tokens;
        
        /* Ensure all indices are different */
        while (chromo->relation_idx == chromo->subject_idx) {
            chromo->relation_idx = rand() % num_tokens;
        }
        while (chromo->object_idx == chromo->subject_idx || chromo->object_idx == chromo->relation_idx) {
            chromo->object_idx = rand() % num_tokens;
        }
        
        chromo->fitness = 0.0f;
    }
    
    ga->generation = 0;
    ga->converged = 0;
}

/* Selection: Tournament selection */
static FactChromosome* select_parent(GeneticAlgorithm* ga) {
    int tournament_size = 3;
    FactChromosome* best = &ga->population[rand() % GA_POPULATION_SIZE];
    
    for (int i = 1; i < tournament_size; i++) {
        FactChromosome* candidate = &ga->population[rand() % GA_POPULATION_SIZE];
        if (candidate->fitness > best->fitness) {
            best = candidate;
        }
    }
    
    return best;
}

/* Crossover: Mix role assignments from two parents */
static FactChromosome crossover(FactChromosome* parent1, FactChromosome* parent2) {
    FactChromosome child;
    
    /* Randomly inherit each role from either parent */
    child.subject_idx = (rand() % 2) ? parent1->subject_idx : parent2->subject_idx;
    child.relation_idx = (rand() % 2) ? parent1->relation_idx : parent2->relation_idx;
    child.object_idx = (rand() % 2) ? parent1->object_idx : parent2->object_idx;
    child.fitness = 0.0f;
    
    return child;
}

/* Mutation: Randomly change one role assignment */
static void mutate(FactChromosome* chromo, int num_tokens) {
    if ((float)rand() / RAND_MAX < GA_MUTATION_RATE) {
        int role = rand() % 3;
        switch (role) {
            case 0: chromo->subject_idx = rand() % num_tokens; break;
            case 1: chromo->relation_idx = rand() % num_tokens; break;
            case 2: chromo->object_idx = rand() % num_tokens; break;
        }
    }
}

/* Run genetic algorithm to discover fact roles */
static GAResult run_genetic_algorithm(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    GAResult result = {0};
    
    if (num_tokens != 3) {
        return result;  /* Only handle 3-token facts for now */
    }
    
    GeneticAlgorithm ga;
    init_population(&ga, num_tokens);
    
    for (int gen = 0; gen < GA_MAX_GENERATIONS; gen++) {
        /* Evaluate fitness */
        float max_fitness = 0.0f;
        for (int i = 0; i < GA_POPULATION_SIZE; i++) {
            ga.population[i].fitness = calculate_fitness(model, &ga.population[i], tokens, num_tokens);
            if (ga.population[i].fitness > max_fitness) {
                max_fitness = ga.population[i].fitness;
                ga.best_solution = ga.population[i];
            }
        }
        
        /* Check convergence */
        int consensus_count = 0;
        for (int i = 0; i < GA_POPULATION_SIZE; i++) {
            if (ga.population[i].fitness > max_fitness * 0.9f) {
                consensus_count++;
            }
        }
        
        if (consensus_count >= GA_POPULATION_SIZE * 0.8f) {
            ga.converged = 1;
            break;
        }
        
        /* Create next generation */
        FactChromosome new_population[GA_POPULATION_SIZE];
        
        for (int i = 0; i < GA_POPULATION_SIZE; i++) {
            FactChromosome* parent1 = select_parent(&ga);
            FactChromosome* parent2 = select_parent(&ga);
            
            new_population[i] = crossover(parent1, parent2);
            mutate(&new_population[i], num_tokens);
        }
        
        /* Replace old population */
        memcpy(ga.population, new_population, sizeof(new_population));
        ga.generation++;
    }
    
    result.converged = ga.converged;
    result.subject_role = ga.best_solution.subject_idx;
    result.relation_role = ga.best_solution.relation_idx;
    result.object_role = ga.best_solution.object_idx;
    result.confidence = ga.best_solution.fitness;
    
    return result;
}

/* =============================================================================
 * SQLITE DATABASE OPERATIONS
 * =============================================================================
 */

/* Initialize SQLite database with proper schema */
static int init_sqlite_db(sqlite3* db) {
    const char* schema_sql = 
        "CREATE TABLE IF NOT EXISTS heuristic_rules ("
        "  pattern_hash TEXT PRIMARY KEY,"
        "  token_pattern TEXT NOT NULL,"
        "  subject_role INTEGER NOT NULL,"
        "  relation_role INTEGER NOT NULL,"
        "  object_role INTEGER NOT NULL,"
        "  confidence REAL NOT NULL,"
        "  hit_count INTEGER DEFAULT 0,"
        "  peak_confidence REAL DEFAULT 0.0,"
        "  created_time INTEGER DEFAULT (strftime('%s', 'now')),"
        "  last_validated INTEGER DEFAULT (strftime('%s', 'now')),"
        "  last_above_threshold INTEGER DEFAULT (strftime('%s', 'now')),"
        "  consecutive_months_irrelevant INTEGER DEFAULT 0"
        ");"
        
        "CREATE INDEX IF NOT EXISTS idx_pattern_hash ON heuristic_rules(pattern_hash);"
        "CREATE INDEX IF NOT EXISTS idx_confidence ON heuristic_rules(confidence);"
        "CREATE INDEX IF NOT EXISTS idx_last_above_threshold ON heuristic_rules(last_above_threshold);"
        
        /* Performance optimizations */
        "PRAGMA journal_mode = WAL;"
        "PRAGMA synchronous = NORMAL;"
        "PRAGMA cache_size = 100000;"
        "PRAGMA auto_vacuum = INCREMENTAL;";
    
    char* error_msg = NULL;
    int result = sqlite3_exec(db, schema_sql, NULL, NULL, &error_msg);
    
    if (result != SQLITE_OK) {
        fprintf(stderr, "❌ SQLite schema error: %s\n", error_msg);
        sqlite3_free(error_msg);
        return -1;
    }
    
    return 0;
}

/* Generate pattern key for caching */
static void generate_pattern_key(char tokens[][MAX_TOKEN_LEN], int num_tokens, char* key, size_t key_size) {
    snprintf(key, key_size, "TOKENS_%d", num_tokens);
    
    /* For now, simple key based on token count */
    /* Later could add POS tags, semantic categories, etc. */
}

/* Lookup cached rule */
static HeuristicRule* lookup_cached_rule(HeuristicCompiler* compiler, const char* pattern_key) {
    if (!compiler->db) return NULL;
    
    const char* sql = "SELECT subject_role, relation_role, object_role, confidence, hit_count "
                     "FROM heuristic_rules WHERE pattern_hash = ? AND confidence >= ?";
    
    sqlite3_stmt* stmt;
    int result = sqlite3_prepare_v2(compiler->db, sql, -1, &stmt, NULL);
    if (result != SQLITE_OK) return NULL;
    
    sqlite3_bind_text(stmt, 1, pattern_key, -1, SQLITE_STATIC);
    sqlite3_bind_double(stmt, 2, RELEVANCY_THRESHOLD);
    
    static HeuristicRule rule;  /* Static storage for return */
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        rule.subject_role = sqlite3_column_int(stmt, 0);
        rule.relation_role = sqlite3_column_int(stmt, 1);
        rule.object_role = sqlite3_column_int(stmt, 2);
        rule.confidence = sqlite3_column_double(stmt, 3);
        rule.hit_count = sqlite3_column_int(stmt, 4);
        strncpy(rule.pattern_hash, pattern_key, sizeof(rule.pattern_hash) - 1);
        rule.pattern_hash[sizeof(rule.pattern_hash) - 1] = '\0';
        
        sqlite3_finalize(stmt);
        return &rule;
    }
    
    sqlite3_finalize(stmt);
    return NULL;
}

/* Cache a new rule */
static void cache_rule(HeuristicCompiler* compiler, const HeuristicRule* rule) {
    if (!compiler->db) return;
    
    const char* sql = "INSERT OR REPLACE INTO heuristic_rules "
                     "(pattern_hash, token_pattern, subject_role, relation_role, object_role, "
                     " confidence, hit_count, peak_confidence, last_validated, last_above_threshold) "
                     "VALUES (?, ?, ?, ?, ?, ?, 1, ?, strftime('%s', 'now'), strftime('%s', 'now'))";
    
    sqlite3_stmt* stmt;
    int result = sqlite3_prepare_v2(compiler->db, sql, -1, &stmt, NULL);
    if (result != SQLITE_OK) return;
    
    sqlite3_bind_text(stmt, 1, rule->pattern_hash, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, rule->token_pattern, -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 3, rule->subject_role);
    sqlite3_bind_int(stmt, 4, rule->relation_role);
    sqlite3_bind_int(stmt, 5, rule->object_role);
    sqlite3_bind_double(stmt, 6, rule->confidence);
    sqlite3_bind_double(stmt, 7, rule->confidence);  /* Initial peak = current */
    
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

/* Update rule usage statistics */
static void update_rule_usage(HeuristicCompiler* compiler, const char* pattern_hash) {
    if (!compiler->db) return;
    
    const char* sql = "UPDATE heuristic_rules SET "
                     "hit_count = hit_count + 1, "
                     "last_validated = strftime('%s', 'now'), "
                     "last_above_threshold = CASE WHEN confidence >= ? THEN strftime('%s', 'now') ELSE last_above_threshold END "
                     "WHERE pattern_hash = ?";
    
    sqlite3_stmt* stmt;
    int result = sqlite3_prepare_v2(compiler->db, sql, -1, &stmt, NULL);
    if (result != SQLITE_OK) return;
    
    sqlite3_bind_double(stmt, 1, RELEVANCY_THRESHOLD);
    sqlite3_bind_text(stmt, 2, pattern_hash, -1, SQLITE_STATIC);
    
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

/* =============================================================================
 * 19-YEAR KNOWLEDGE PRESERVATION SYSTEM
 * =============================================================================
 */

/* Calculate relevancy score for a rule */
static float calculate_relevancy_score(const HeuristicRule* rule) {
    time_t now = time(NULL);
    time_t age_days = (now - rule->created_time) / (24 * 3600);
    
    if (age_days == 0) age_days = 1;  /* Avoid division by zero */
    
    float recency_factor = (float)rule->hit_count / age_days;
    float confidence_factor = rule->confidence;
    float peak_factor = rule->peak_confidence * 0.3f;
    
    return (recency_factor * 0.5f) + (confidence_factor * 0.3f) + (peak_factor * 0.2f);
}

/* Ultra-conservative cleanup: only the most irrelevant rule after 19 years */
static void cleanup_persistently_irrelevant_rules(HeuristicCompiler* compiler) {
    if (!compiler->db) return;
    
    time_t nineteen_years = 19LL * 365 * 24 * 3600;
    time_t current_time = time(NULL);
    
    const char* find_sql = 
        "SELECT pattern_hash, confidence, hit_count, created_time "
        "FROM heuristic_rules WHERE "
        "confidence < ? AND "
        "hit_count < ? AND "
        "last_above_threshold < ? AND "
        "last_above_threshold > 0 "
        "ORDER BY (confidence * hit_count) ASC "
        "LIMIT 1";
    
    sqlite3_stmt* stmt;
    int result = sqlite3_prepare_v2(compiler->db, find_sql, -1, &stmt, NULL);
    if (result != SQLITE_OK) return;
    
    sqlite3_bind_double(stmt, 1, RELEVANCY_THRESHOLD);
    sqlite3_bind_int(stmt, 2, MIN_HIT_COUNT);
    sqlite3_bind_int64(stmt, 3, current_time - nineteen_years);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* pattern_hash = (const char*)sqlite3_column_text(stmt, 0);
        float confidence = sqlite3_column_double(stmt, 1);
        int hit_count = sqlite3_column_int(stmt, 2);
        
        printf("🗑️  Found rule eligible for 19-year cleanup:\n");
        printf("    Pattern: %s (confidence: %.3f, hits: %d)\n", pattern_hash, confidence, hit_count);
        printf("    This is the LEAST relevant rule in the entire knowledge vault\n");
        
        /* Delete the single most irrelevant rule */
        const char* delete_sql = "DELETE FROM heuristic_rules WHERE pattern_hash = ?";
        sqlite3_stmt* delete_stmt;
        sqlite3_prepare_v2(compiler->db, delete_sql, -1, &delete_stmt, NULL);
        sqlite3_bind_text(delete_stmt, 1, pattern_hash, -1, SQLITE_STATIC);
        sqlite3_step(delete_stmt);
        sqlite3_finalize(delete_stmt);
        
        printf("✨ One persistently irrelevant rule removed after 19+ years\n");
        compiler->stats.rules_cleaned_19_year++;
    } else {
        printf("🏛️  No rules qualify for 19-year cleanup - all knowledge preserved\n");
    }
    
    sqlite3_finalize(stmt);
}

/* Annual audit and potential cleanup */
static void annual_relevancy_audit(HeuristicCompiler* compiler) {
    static time_t last_audit = 0;
    time_t now = time(NULL);
    
    if (now - last_audit > 365 * 24 * 3600) {  /* Once per year */
        printf("📊 === Annual Knowledge Vault Audit ===\n");
        
        cleanup_persistently_irrelevant_rules(compiler);
        
        /* Print statistics */
        const char* stats_sql = "SELECT COUNT(*), AVG(confidence), AVG(hit_count) FROM heuristic_rules";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(compiler->db, stats_sql, -1, &stmt, NULL);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            int total_rules = sqlite3_column_int(stmt, 0);
            double avg_confidence = sqlite3_column_double(stmt, 1);
            double avg_hits = sqlite3_column_double(stmt, 2);
            
            printf("📈 Knowledge vault status:\n");
            printf("   Total rules: %d\n", total_rules);
            printf("   Average confidence: %.3f\n", avg_confidence);
            printf("   Average hit count: %.1f\n", avg_hits);
            printf("   Rules cleaned in 19+ years: %llu\n", compiler->stats.rules_cleaned_19_year);
        }
        
        sqlite3_finalize(stmt);
        last_audit = now;
        printf("✅ Annual audit complete\n\n");
    }
}

/* =============================================================================
 * PUBLIC API IMPLEMENTATION
 * =============================================================================
 */

/* Create heuristic compiler instance */
HeuristicCompiler* create_heuristic_compiler(const char* db_path) {
    HeuristicCompiler* compiler = calloc(1, sizeof(HeuristicCompiler));
    if (!compiler) {
        fprintf(stderr, "❌ Failed to allocate heuristic compiler\n");
        return NULL;
    }
    
    /* Open SQLite database */
    int result = sqlite3_open(db_path, &compiler->db);
    if (result != SQLITE_OK) {
        fprintf(stderr, "❌ Failed to open SQLite database: %s\n", sqlite3_errmsg(compiler->db));
        free(compiler);
        return NULL;
    }
    
    /* Initialize database schema */
    if (init_sqlite_db(compiler->db) != 0) {
        sqlite3_close(compiler->db);
        free(compiler);
        return NULL;
    }
    
    /* Initialize statistics */
    compiler->stats.system_birth_time = time(NULL);
    
    printf("🧬 Heuristic compiler initialized with knowledge vault: %s\n", db_path);
    return compiler;
}

/* Destroy heuristic compiler */
void destroy_heuristic_compiler(HeuristicCompiler* compiler) {
    if (!compiler) return;
    
    printf("📊 Heuristic compiler final statistics:\n");
    printf("   Cache hits: %llu\n", compiler->stats.total_cache_hits);
    printf("   GA discoveries: %llu\n", compiler->stats.total_discoveries);
    printf("   Rules cleaned (19yr): %llu\n", compiler->stats.rules_cleaned_19_year);
    
    if (compiler->db) {
        sqlite3_close(compiler->db);
    }
    
    free(compiler);
}

/* Main fact extraction with hybrid GA-cache approach */
int extract_facts_with_heuristic_compiler(GalileoModel* model, char tokens[][MAX_TOKEN_LEN], int num_tokens) {
    if (!heuristic_module_initialized || !g_compiler) {
        /* Fallback: module not loaded */
        return 0;
    }
    
#if GALILEO_HAS_PTHREAD
    pthread_mutex_lock(&compiler_mutex);
#endif
    
    int facts_extracted = 0;
    
    /* Generate pattern key */
    char pattern_key[256];
    generate_pattern_key(tokens, num_tokens, pattern_key, sizeof(pattern_key));
    
    /* Try cache first (microsecond lookup) */
    HeuristicRule* cached_rule = lookup_cached_rule(g_compiler, pattern_key);
    if (cached_rule && cached_rule->confidence >= CONFIDENCE_THRESHOLD) {
        /* Apply cached rule instantly */
        if (cached_rule->subject_role < num_tokens && 
            cached_rule->relation_role < num_tokens && 
            cached_rule->object_role < num_tokens) {
            
            /* Extract fact using cached role assignment */
            galileo_add_enhanced_fact_safe(model, 
                tokens[cached_rule->subject_role], 
                tokens[cached_rule->relation_role], 
                tokens[cached_rule->object_role], 
                cached_rule->confidence, NULL, 0);
            
            /* Update usage statistics */
            update_rule_usage(g_compiler, pattern_key);
            g_compiler->stats.total_cache_hits++;
            facts_extracted = 1;
            
            printf("⚡ Cache hit: %s %s %s (%.2f confidence)\n",
                   tokens[cached_rule->subject_role],
                   tokens[cached_rule->relation_role], 
                   tokens[cached_rule->object_role],
                   cached_rule->confidence);
        }
    } else {
        /* Cache miss: Run genetic algorithm discovery */
        GAResult discovery = run_genetic_algorithm(model, tokens, num_tokens);
        
        if (discovery.converged && discovery.confidence >= CONFIDENCE_THRESHOLD) {
            /* Extract fact using GA discovery */
            galileo_add_enhanced_fact_safe(model,
                tokens[discovery.subject_role],
                tokens[discovery.relation_role], 
                tokens[discovery.object_role],
                discovery.confidence, NULL, 0);
            
            /* Compile and cache the new rule */
            HeuristicRule new_rule = {0};
            strncpy(new_rule.pattern_hash, pattern_key, sizeof(new_rule.pattern_hash) - 1);
            snprintf(new_rule.token_pattern, sizeof(new_rule.token_pattern), 
                    "TOKENS_%d", num_tokens);
            new_rule.subject_role = discovery.subject_role;
            new_rule.relation_role = discovery.relation_role;
            new_rule.object_role = discovery.object_role;
            new_rule.confidence = discovery.confidence;
            new_rule.created_time = time(NULL);
            
            cache_rule(g_compiler, &new_rule);
            g_compiler->stats.total_discoveries++;
            facts_extracted = 1;
            
            printf("🧬 GA discovery: %s %s %s (%.2f confidence) - CACHED\n",
                   tokens[discovery.subject_role],
                   tokens[discovery.relation_role],
                   tokens[discovery.object_role], 
                   discovery.confidence);
        }
    }
    
    /* Periodic annual audit */
    annual_relevancy_audit(g_compiler);
    
#if GALILEO_HAS_PTHREAD
    pthread_mutex_unlock(&compiler_mutex);
#endif
    
    return facts_extracted;
}

/* Get compiler statistics */
HeuristicStats get_heuristic_compiler_stats(void) {
    HeuristicStats empty_stats = {0};
    
    if (!heuristic_module_initialized || !g_compiler) {
        return empty_stats;
    }
    
    return g_compiler->stats;
}
