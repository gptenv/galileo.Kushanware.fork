/* =============================================================================
 * galileo/src/symbolic/galileo_symbolic.c - Symbolic Reasoning Engine
 * 
 * Hot-loadable shared library implementing symbolic logic, multi-hop inference,
 * exception handling, quantified reasoning, and conflict resolution.
 * 
 * This is where Galileo transcends pure neural computation and gains the ability
 * to perform logical reasoning, handle exceptions ("penguins can't fly"), and
 * resolve conflicts through symbolic manipulation.
 * 
 * Extracted from galileo_legacy_core-v42-v3.pre-modular.best.c with enhanced
 * safety, depth limiting, and conflict resolution capabilities.
 * =============================================================================
 */

#include "galileo_symbolic.h"
#include "../core/galileo_core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdarg.h>
#include <stddef.h>

/* =============================================================================
 * MODULE METADATA AND INITIALIZATION
 * =============================================================================
 */

static int symbolic_module_initialized = 0;
static int reasoning_depth = 0;  /* Thread-local reasoning depth counter */

/* Module initialization */
static int symbolic_module_init(void) {
    if (symbolic_module_initialized) {
        return 0;  /* Already initialized */
    }
    
    fprintf(stderr, "🧠 Symbolic reasoning module v42.1 initializing...\n");
    
    /* Initialize reasoning depth counter */
    reasoning_depth = 0;
    
    symbolic_module_initialized = 1;
    fprintf(stderr, "✅ Symbolic reasoning ready! Logic engine online.\n");
    return 0;
}

/* Module cleanup */
static void symbolic_module_cleanup(void) {
    if (!symbolic_module_initialized) {
        return;
    }
    
    fprintf(stderr, "🧠 Symbolic reasoning module shutting down...\n");
    symbolic_module_initialized = 0;
}

/* Module info structure for dynamic loading */
SymbolicModuleInfo symbolic_module_info = {
    .name = "symbolic",
    .version = "42.1",
    .init_func = symbolic_module_init,
    .cleanup_func = symbolic_module_cleanup
};

/* =============================================================================
 * DYNAMIC BUFFER MANAGEMENT FOR CONFLICT DESCRIPTIONS
 * Computer science-y algorithms for smart memory management, eh!
 * =============================================================================
 */

/* Initialize dynamic conflict description */
static DynamicConflictDescription* create_dynamic_conflict_description(void) {
    DynamicConflictDescription* desc = malloc(sizeof(DynamicConflictDescription));
    if (!desc) return NULL;
    
    desc->capacity = CONFLICT_DESCRIPTION_INITIAL_SIZE;
    desc->data = malloc(desc->capacity);
    if (!desc->data) {
        free(desc);
        return NULL;
    }
    
    desc->size = 0;
    desc->data[0] = '\0';
    
    #ifdef MAX_CONFLICT_DESCRIPTION_LIMIT
    desc->max_capacity = MAX_CONFLICT_DESCRIPTION_LIMIT;
    #else
    desc->max_capacity = SIZE_MAX;  /* No limit */
    #endif
    
    return desc;
}

/* Free dynamic conflict description */
static void free_dynamic_conflict_description(DynamicConflictDescription* desc) {
    if (desc) {
        if (desc->data) free(desc->data);
        free(desc);
    }
}

/* Grow buffer using exponential growth strategy */
static int grow_conflict_buffer(DynamicConflictDescription* desc, size_t needed_size) {
    if (!desc) return 0;
    
    /* Calculate new capacity using exponential growth */
    size_t new_capacity = desc->capacity;
    while (new_capacity < needed_size) {
        new_capacity *= CONFLICT_DESCRIPTION_GROWTH_FACTOR;
        
        /* Check against maximum limit */
        if (new_capacity > desc->max_capacity) {
            new_capacity = desc->max_capacity;
            if (new_capacity < needed_size) {
                return 0;  /* Cannot satisfy request within limits */
            }
            break;
        }
    }
    
    /* Reallocate buffer */
    char* new_data = realloc(desc->data, new_capacity);
    if (!new_data) return 0;
    
    desc->data = new_data;
    desc->capacity = new_capacity;
    return 1;
}

/* Shrink buffer using hysteresis to prevent thrashing */
static void maybe_shrink_conflict_buffer(DynamicConflictDescription* desc) {
    if (!desc || desc->capacity <= CONFLICT_DESCRIPTION_INITIAL_SIZE) return;
    
    /* Only shrink if we're using less than 1/4 of capacity (hysteresis) */
    if (desc->size * CONFLICT_DESCRIPTION_SHRINK_THRESHOLD < desc->capacity) {
        size_t new_capacity = desc->capacity / 2;
        
        /* Don't shrink below initial size */
        if (new_capacity < CONFLICT_DESCRIPTION_INITIAL_SIZE) {
            new_capacity = CONFLICT_DESCRIPTION_INITIAL_SIZE;
        }
        
        /* Don't shrink if it would make buffer too small for current content */
        if (new_capacity < desc->size + 1) return;
        
        char* new_data = realloc(desc->data, new_capacity);
        if (new_data) {  /* Only shrink if realloc succeeds */
            desc->data = new_data;
            desc->capacity = new_capacity;
        }
    }
}

/* Add conflict description with dynamic allocation */
static int add_conflict_description(GalileoModel* model, const char* format, ...) {
    if (!model || !format) return 0;
    
    /* Grow conflicts array if needed */
    if (model->num_resolved_conflicts >= model->resolved_conflicts_capacity) {
        int new_capacity = model->resolved_conflicts_capacity * 2;
        DynamicConflictDescription** new_array = realloc(model->resolved_conflicts, 
                                                         new_capacity * sizeof(DynamicConflictDescription*));
        if (!new_array) return 0;
        
        model->resolved_conflicts = new_array;
        model->resolved_conflicts_capacity = new_capacity;
        
        /* Initialize new slots to NULL */
        for (int i = model->num_resolved_conflicts; i < new_capacity; i++) {
            model->resolved_conflicts[i] = NULL;
        }
    }
    
    /* Create new dynamic description */
    DynamicConflictDescription* desc = create_dynamic_conflict_description();
    if (!desc) return 0;
    
    /* Format the conflict description */
    va_list args;
    va_start(args, format);
    
    va_list args_copy;
    va_copy(args_copy, args);
    int needed = vsnprintf(NULL, 0, format, args);
    va_end(args);
    
    if (needed < 0) {
        va_end(args_copy);
        free_dynamic_conflict_description(desc);
        return 0;
    }
    
    size_t needed_size = (size_t)needed + 1;
    if (needed_size > desc->capacity && !grow_conflict_buffer(desc, needed_size)) {
        va_end(args_copy);
        free_dynamic_conflict_description(desc);
        return 0;
    }
    
    vsnprintf(desc->data, desc->capacity, format, args_copy);
    va_end(args_copy);
    desc->size = (size_t)needed;
    
    /* Try to shrink buffer if it's gotten too big */
    maybe_shrink_conflict_buffer(desc);
    
    /* Add to model */
    model->resolved_conflicts[model->num_resolved_conflicts] = desc;
    model->num_resolved_conflicts++;
    
    return 1;
}

/* Normalize string for comparison (lowercase, trim) */
static void normalize_string(const char* input, char* output, size_t max_len) {
    if (!input || !output || max_len == 0) return;
    
    size_t len = strlen(input);
    size_t out_idx = 0;
    
    /* Skip leading whitespace */
    size_t start = 0;
    while (start < len && isspace(input[start])) start++;
    
    /* Copy and convert to lowercase */
    for (size_t i = start; i < len && out_idx < max_len - 1; i++) {
        if (!isspace(input[i]) || (out_idx > 0 && output[out_idx-1] != ' ')) {
            output[out_idx++] = tolower(input[i]);
        }
    }
    
    /* Remove trailing whitespace */
    while (out_idx > 0 && isspace(output[out_idx-1])) out_idx--;
    
    output[out_idx] = '\0';
}

/* Check if two normalized strings are equivalent */
static int strings_equivalent(const char* str1, const char* str2) {
    char norm1[256], norm2[256];
    normalize_string(str1, norm1, sizeof(norm1));
    normalize_string(str2, norm2, sizeof(norm2));
    return strcmp(norm1, norm2) == 0;
}

/* Extract numeric value from string for arithmetic reasoning */
static float extract_numeric_value(const char* str) {
    if (!str) return 0.0f;
    
    /* Simple numeric extraction - look for numbers in the string */
    char* endptr;
    float value = strtof(str, &endptr);
    
    /* If no valid number found, try to extract from common patterns */
    if (endptr == str) {
        if (strstr(str, "zero")) return 0.0f;
        if (strstr(str, "one")) return 1.0f;
        if (strstr(str, "two")) return 2.0f;
        if (strstr(str, "three")) return 3.0f;
        if (strstr(str, "four")) return 4.0f;
        if (strstr(str, "five")) return 5.0f;
        return 0.0f;  /* Default */
    }
    
    return value;
}

/* =============================================================================
 * FACT MANAGEMENT WITH DEDUPLICATION
 * =============================================================================
 */

/* Check if fact already exists - PHASE 0 enhancement */
int fact_exists(GalileoModel* model, const char* subject, const char* relation, const char* object) {
    if (!model || !subject || !relation || !object) return 0;
    
    for (int i = 0; i < model->num_facts; i++) {
        SymbolicFact* fact = &model->facts[i];
        if (strings_equivalent(fact->subject, subject) &&
            strings_equivalent(fact->relation, relation) &&
            strings_equivalent(fact->object, object)) {
            return 1;  /* Found duplicate */
        }
    }
    return 0;  /* Not found */
}

/* Enhanced fact addition with metadata and deduplication - PHASE 0 enhancement */
void galileo_add_enhanced_fact_safe(GalileoModel* model, const char* subject, const char* relation, 
                                   const char* object, float confidence, int* supporting_nodes, int support_count) {
    if (!model || !subject || !relation || !object) return;
    
    /* Ensure symbolic module is initialized */
    if (!symbolic_module_initialized) {
        symbolic_module_init();
    }
    
    if (model->num_facts >= MAX_FACTS) {
        printf("⚠️  Maximum facts limit reached (%d)\n", MAX_FACTS);
        return;
    }
    
    /* Check for duplicate */
    if (fact_exists(model, subject, relation, object)) {
        /* Update confidence if higher */
        for (int i = 0; i < model->num_facts; i++) {
            SymbolicFact* fact = &model->facts[i];
            if (strings_equivalent(fact->subject, subject) &&
                strings_equivalent(fact->relation, relation) &&
                strings_equivalent(fact->object, object)) {
                if (confidence > fact->confidence) {
                    fact->confidence = confidence;
                    printf("📝 Updated fact confidence: %s %s %s (%.2f)\n", 
                           subject, relation, object, confidence);
                }
                return;
            }
        }
    }
    
    /* Add new fact */
    SymbolicFact* fact = &model->facts[model->num_facts];
    
    /* Copy strings with bounds checking */
    strncpy(fact->subject, subject, sizeof(fact->subject) - 1);
    fact->subject[sizeof(fact->subject) - 1] = '\0';
    
    strncpy(fact->relation, relation, sizeof(fact->relation) - 1);
    fact->relation[sizeof(fact->relation) - 1] = '\0';
    
    strncpy(fact->object, object, sizeof(fact->object) - 1);
    fact->object[sizeof(fact->object) - 1] = '\0';
    
    /* Set metadata */
    fact->confidence = fmaxf(0.0f, fminf(1.0f, confidence));  /* Clamp to [0,1] */
    fact->derived = (support_count > 0) ? 1 : 0;
    fact->iteration_added = model->current_iteration;
    
    /* Copy supporting nodes */
    int nodes_to_copy = (support_count > MAX_SUPPORTING_NODES) ? MAX_SUPPORTING_NODES : support_count;
    for (int i = 0; i < nodes_to_copy; i++) {
        fact->supporting_nodes[i] = supporting_nodes[i];
    }
    fact->support_count = nodes_to_copy;
    
    model->num_facts++;
    
    printf("💡 Added fact: %s %s %s (conf: %.2f, support: %d)\n", 
           subject, relation, object, confidence, support_count);
}

/* Legacy compatibility wrapper */
void galileo_add_fact(GalileoModel* model, const char* subject, const char* relation, const char* object, float confidence) {
    galileo_add_enhanced_fact_safe(model, subject, relation, object, confidence, NULL, 0);
}

/* =============================================================================
 * SYMBOLIC REASONING PATTERNS
 * =============================================================================
 */

/* Pattern 1: Transitivity reasoning (A->B, B->C => A->C) */
static void apply_transitivity_reasoning(GalileoModel* model) {
    int facts_added = 0;
    
    for (int i = 0; i < model->num_facts; i++) {
        for (int j = 0; j < model->num_facts; j++) {
            if (i == j) continue;
            
            SymbolicFact* fact1 = &model->facts[i];
            SymbolicFact* fact2 = &model->facts[j];
            
            /* Check for transitivity pattern: A rel B, B rel C => A rel C */
            if (strings_equivalent(fact1->object, fact2->subject) &&
                strings_equivalent(fact1->relation, fact2->relation)) {
                
                /* Avoid infinite loops and low-confidence chains */
                if (fact1->confidence > 0.3f && fact2->confidence > 0.3f &&
                    !strings_equivalent(fact1->subject, fact2->object)) {
                    
                    /* Check if conclusion already exists */
                    if (!fact_exists(model, fact1->subject, fact1->relation, fact2->object)) {
                        float new_confidence = fact1->confidence * fact2->confidence * 0.8f;  /* Decay */
                        
                        int supporting_nodes[] = {fact1->supporting_nodes[0], fact2->supporting_nodes[0]};
                        galileo_add_enhanced_fact_safe(model, fact1->subject, fact1->relation, 
                                                      fact2->object, new_confidence, supporting_nodes, 2);
                        facts_added++;
                        
                        printf("🔗 Transitivity: %s %s %s + %s %s %s => %s %s %s\n",
                               fact1->subject, fact1->relation, fact1->object,
                               fact2->subject, fact2->relation, fact2->object,
                               fact1->subject, fact1->relation, fact2->object);
                    }
                }
            }
        }
    }
    
    if (facts_added > 0) {
        printf("✨ Applied transitivity reasoning: %d new facts\n", facts_added);
    }
}

/* Pattern 2: Subclass inheritance with exceptions */
static void apply_inheritance_reasoning(GalileoModel* model) {
    int facts_added = 0;
    
    for (int i = 0; i < model->num_facts; i++) {
        for (int j = 0; j < model->num_facts; j++) {
            if (i == j) continue;
            
            SymbolicFact* subclass_fact = &model->facts[i];
            SymbolicFact* property_fact = &model->facts[j];
            
            /* Look for: X subclass_of Y, Y has_property Z => X has_property Z */
            if ((strings_equivalent(subclass_fact->relation, "subclass_of") || 
                 strings_equivalent(subclass_fact->relation, "is_a") ||
                 strings_equivalent(subclass_fact->relation, "type_of")) &&
                (strings_equivalent(property_fact->relation, "can") ||
                 strings_equivalent(property_fact->relation, "has") ||
                 strings_equivalent(property_fact->relation, "is")) &&
                strings_equivalent(subclass_fact->object, property_fact->subject)) {
                
                /* Check for explicit exceptions first */
                int has_exception = 0;
                for (int k = 0; k < model->num_facts; k++) {
                    SymbolicFact* exception = &model->facts[k];
                    if ((strings_equivalent(exception->relation, "cannot") ||
                         strings_equivalent(exception->relation, "does_not") ||
                         strings_equivalent(exception->relation, "not")) &&
                        strings_equivalent(exception->subject, subclass_fact->subject) &&
                        strings_equivalent(exception->object, property_fact->object)) {
                        has_exception = 1;
                        printf("🚫 Exception found: %s %s %s (blocks inheritance)\n",
                               exception->subject, exception->relation, exception->object);
                        break;
                    }
                }
                
                if (!has_exception && subclass_fact->confidence > 0.4f && property_fact->confidence > 0.4f) {
                    /* Check if conclusion already exists */
                    if (!fact_exists(model, subclass_fact->subject, property_fact->relation, property_fact->object)) {
                        float new_confidence = subclass_fact->confidence * property_fact->confidence * 0.9f;
                        
                        int supporting_nodes[] = {subclass_fact->supporting_nodes[0], property_fact->supporting_nodes[0]};
                        galileo_add_enhanced_fact_safe(model, subclass_fact->subject, property_fact->relation, 
                                                      property_fact->object, new_confidence, supporting_nodes, 2);
                        facts_added++;
                        
                        printf("🧬 Inheritance: %s %s %s + %s %s %s => %s %s %s\n",
                               subclass_fact->subject, subclass_fact->relation, subclass_fact->object,
                               property_fact->subject, property_fact->relation, property_fact->object,
                               subclass_fact->subject, property_fact->relation, property_fact->object);
                    }
                }
            }
        }
    }
    
    if (facts_added > 0) {
        printf("✨ Applied inheritance reasoning: %d new facts\n", facts_added);
    }
}

/* Pattern 3: Quantified reasoning ("most", "some", "all") */
static void apply_quantified_reasoning(GalileoModel* model) {
    int facts_added = 0;
    
    for (int i = 0; i < model->num_facts; i++) {
        SymbolicFact* fact = &model->facts[i];
        
        /* Handle quantified statements */
        if (strstr(fact->subject, "most") || strstr(fact->subject, "many") || strstr(fact->subject, "usually")) {
            /* Extract the actual subject */
            char actual_subject[128];
            const char* keywords[] = {"most", "many", "usually", "typically"};
            
            strcpy(actual_subject, fact->subject);
            for (int k = 0; k < 4; k++) {
                char* pos = strstr(actual_subject, keywords[k]);
                if (pos) {
                    /* Remove the quantifier */
                    memmove(pos, pos + strlen(keywords[k]), strlen(pos + strlen(keywords[k])) + 1);
                    while (*pos == ' ') memmove(pos, pos + 1, strlen(pos));
                    break;
                }
            }
            
            /* Add probabilistic fact with reduced confidence */
            if (!fact_exists(model, actual_subject, fact->relation, fact->object)) {
                float new_confidence = fact->confidence * 0.7f;  /* "most" = 70% confidence */
                galileo_add_enhanced_fact_safe(model, actual_subject, fact->relation, fact->object, 
                                              new_confidence, fact->supporting_nodes, fact->support_count);
                facts_added++;
                
                printf("📊 Quantified: %s => %s %s %s (%.2f confidence)\n",
                       fact->subject, actual_subject, fact->relation, fact->object, new_confidence);
            }
        }
    }
    
    if (facts_added > 0) {
        printf("✨ Applied quantified reasoning: %d new facts\n", facts_added);
    }
}

/* Pattern 4: Causal chain reasoning */
static void apply_causal_reasoning(GalileoModel* model) {
    int facts_added = 0;
    
    for (int i = 0; i < model->num_facts; i++) {
        for (int j = 0; j < model->num_facts; j++) {
            if (i == j) continue;
            
            SymbolicFact* cause = &model->facts[i];
            SymbolicFact* effect = &model->facts[j];
            
            /* Look for causal chains: A causes B, B causes C => A influences C */
            if ((strings_equivalent(cause->relation, "causes") ||
                 strings_equivalent(cause->relation, "leads_to") ||
                 strings_equivalent(cause->relation, "results_in")) &&
                (strings_equivalent(effect->relation, "causes") ||
                 strings_equivalent(effect->relation, "leads_to") ||
                 strings_equivalent(effect->relation, "results_in")) &&
                strings_equivalent(cause->object, effect->subject)) {
                
                if (cause->confidence > 0.5f && effect->confidence > 0.5f &&
                    !strings_equivalent(cause->subject, effect->object)) {
                    
                    if (!fact_exists(model, cause->subject, "influences", effect->object)) {
                        float new_confidence = cause->confidence * effect->confidence * 0.6f;  /* Weaker than direct */
                        
                        int supporting_nodes[] = {cause->supporting_nodes[0], effect->supporting_nodes[0]};
                        galileo_add_enhanced_fact_safe(model, cause->subject, "influences", 
                                                      effect->object, new_confidence, supporting_nodes, 2);
                        facts_added++;
                        
                        printf("⚡ Causal chain: %s %s %s + %s %s %s => %s influences %s\n",
                               cause->subject, cause->relation, cause->object,
                               effect->subject, effect->relation, effect->object,
                               cause->subject, effect->object);
                    }
                }
            }
        }
    }
    
    if (facts_added > 0) {
        printf("✨ Applied causal reasoning: %d new facts\n", facts_added);
    }
}

/* Pattern 5: Simple arithmetic reasoning */
static void apply_arithmetic_reasoning(GalileoModel* model) {
    int facts_added = 0;
    
    for (int i = 0; i < model->num_facts; i++) {
        SymbolicFact* fact = &model->facts[i];
        
        /* Look for arithmetic patterns */
        if (strings_equivalent(fact->relation, "equals") || 
            strings_equivalent(fact->relation, "is") ||
            strings_equivalent(fact->relation, "=")) {
            
            float subject_val = extract_numeric_value(fact->subject);
            float object_val = extract_numeric_value(fact->object);
            
            if (subject_val != 0.0f || object_val != 0.0f) {
                /* Simple arithmetic sanity checks */
                char result_subject[128], result_object[128];
                
                /* Generate some basic arithmetic facts */
                if (subject_val > 0 && object_val > 0) {
                    snprintf(result_subject, sizeof(result_subject), "%.1f plus %.1f", subject_val, object_val);
                    snprintf(result_object, sizeof(result_object), "%.1f", subject_val + object_val);
                    
                    if (!fact_exists(model, result_subject, "equals", result_object)) {
                        galileo_add_enhanced_fact_safe(model, result_subject, "equals", result_object, 
                                                      0.95f, fact->supporting_nodes, fact->support_count);
                        facts_added++;
                        
                        printf("🔢 Arithmetic: %.1f + %.1f = %.1f\n", subject_val, object_val, subject_val + object_val);
                    }
                }
            }
        }
    }
    
    if (facts_added > 0) {
        printf("✨ Applied arithmetic reasoning: %d new facts\n", facts_added);
    }
}

/* =============================================================================
 * CONFLICT DETECTION AND RESOLUTION
 * =============================================================================
 */

/* Detect and resolve contradictions */
static void detect_and_resolve_conflicts(GalileoModel* model) {
    int conflicts_found = 0;
    
    for (int i = 0; i < model->num_facts; i++) {
        for (int j = i + 1; j < model->num_facts; j++) {
            SymbolicFact* fact1 = &model->facts[i];
            SymbolicFact* fact2 = &model->facts[j];
            
            /* Look for direct contradictions: X can Y vs X cannot Y */
            if (strings_equivalent(fact1->subject, fact2->subject) &&
                strings_equivalent(fact1->object, fact2->object)) {
                
                int is_contradiction = 0;
                
                /* Check for positive vs negative relations */
                if ((strings_equivalent(fact1->relation, "can") && strings_equivalent(fact2->relation, "cannot")) ||
                    (strings_equivalent(fact1->relation, "is") && strings_equivalent(fact2->relation, "is_not")) ||
                    (strings_equivalent(fact1->relation, "has") && strings_equivalent(fact2->relation, "does_not_have"))) {
                    is_contradiction = 1;
                }
                
                if (is_contradiction) {
                    conflicts_found++;
                    
                    printf("⚠️  Conflict detected: %s %s %s vs %s %s %s\n",
                           fact1->subject, fact1->relation, fact1->object,
                           fact2->subject, fact2->relation, fact2->object);
                    
                    /* Resolve by confidence and specificity */
                    SymbolicFact* winner;
                    SymbolicFact* loser;
                    
                    /* Prefer more specific facts (exceptions) and higher confidence */
                    float specificity1 = strings_equivalent(fact1->relation, "cannot") ? 1.2f : 1.0f;
                    float specificity2 = strings_equivalent(fact2->relation, "cannot") ? 1.2f : 1.0f;
                    
                    float score1 = fact1->confidence * specificity1;
                    float score2 = fact2->confidence * specificity2;
                    
                    if (score1 > score2) {
                        winner = fact1;
                        loser = fact2;
                    } else {
                        winner = fact2;
                        loser = fact1;
                    }
                    
                    /* Record resolution using dynamic buffer system */
                    if (add_conflict_description(model,
                            "Kept: %s %s %s (%.2f) vs %s %s %s (%.2f)",
                            winner->subject, winner->relation, winner->object, winner->confidence,
                            loser->subject, loser->relation, loser->object, loser->confidence)) {
                        /* Success - conflict recorded with dynamic sizing */
                    } else {
                        printf("⚠️  Warning: Could not record conflict resolution (memory allocation failed)\n");
                    }
                    
                    /* Lower confidence of the losing fact */
                    loser->confidence *= 0.3f;
                    
                    printf("🏆 Resolved: Kept %s %s %s (%.2f), weakened %s %s %s (%.2f)\n",
                           winner->subject, winner->relation, winner->object, winner->confidence,
                           loser->subject, loser->relation, loser->object, loser->confidence);
                }
            }
        }
    }
    
    if (conflicts_found > 0) {
        printf("⚖️  Resolved %d conflicts\n", conflicts_found);
    }
}

/* =============================================================================
 * MAIN SYMBOLIC INFERENCE ENGINE
 * =============================================================================
 */

/* Enhanced symbolic inference with safety and depth limiting - PHASE 0 enhancement */
void galileo_enhanced_symbolic_inference_safe(GalileoModel* model) {
    if (!model) return;
    
    /* Ensure symbolic module is initialized */
    if (!symbolic_module_initialized) {
        symbolic_module_init();
    }
    
    /* Prevent infinite recursion with depth limiting */
    if (reasoning_depth >= 10) {
        printf("🔄 Maximum reasoning depth reached, stopping inference\n");
        return;
    }
    
    reasoning_depth++;
    model->total_symbolic_calls++;
    
    printf("\n🧠 === Symbolic Inference (depth %d) ===\n", reasoning_depth);
    
    if (model->num_facts == 0) {
        printf("💭 No facts to reason about\n");
        reasoning_depth--;
        return;
    }
    
    printf("📚 Starting with %d facts...\n", model->num_facts);
    
    int initial_facts = model->num_facts;
    
    /* Apply reasoning patterns in order */
    apply_transitivity_reasoning(model);
    apply_inheritance_reasoning(model);
    apply_quantified_reasoning(model);
    apply_causal_reasoning(model);
    apply_arithmetic_reasoning(model);
    
    /* Detect and resolve conflicts */
    detect_and_resolve_conflicts(model);
    
    int facts_added = model->num_facts - initial_facts;
    
    printf("✨ Symbolic inference complete: %d new facts derived\n", facts_added);
    printf("📊 Total facts: %d, conflicts resolved: %d\n", 
           model->num_facts, model->num_resolved_conflicts);
    
    /* Periodic memory cleanup - shrink oversized conflict buffers */
    for (int i = 0; i < model->num_resolved_conflicts; i++) {
        if (model->resolved_conflicts[i]) {
            maybe_shrink_conflict_buffer(model->resolved_conflicts[i]);
        }
    }
    
    /* Recursive application if new facts were derived (with depth limit) */
    if (facts_added > 0 && reasoning_depth < 5) {
        printf("🔄 New facts found, applying recursive inference...\n");
        galileo_enhanced_symbolic_inference_safe(model);
    }
    
    reasoning_depth--;
}

/* Legacy compatibility wrapper */
void galileo_symbolic_inference(GalileoModel* model) {
    galileo_enhanced_symbolic_inference_safe(model);
}
