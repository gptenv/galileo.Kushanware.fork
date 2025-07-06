/* =============================================================================
 * galileo/src/symbolic/galileo_symbolic.h - Symbolic Reasoning Module Public API
 * 
 * Public header for the symbolic reasoning module containing all functions
 * for logical inference, fact management, exception handling, and conflict
 * resolution.
 * 
 * This module gives Galileo the ability to perform logical reasoning beyond
 * pure neural computation - handling transitivity, inheritance with exceptions,
 * quantified reasoning, and symbolic manipulation.
 * =============================================================================
 */

#ifndef GALILEO_SYMBOLIC_H
#define GALILEO_SYMBOLIC_H

#include "../core/galileo_types.h"

/* =============================================================================
 * FACT MANAGEMENT FUNCTIONS
 * =============================================================================
 */

/* Enhanced fact addition with metadata and deduplication */
void galileo_add_enhanced_fact_safe(GalileoModel* model, 
                                   const char* subject, 
                                   const char* relation, 
                                   const char* object, 
                                   float confidence, 
                                   int* supporting_nodes, 
                                   int support_count);

/* Fact existence checking for deduplication */
int fact_exists(GalileoModel* model, 
               const char* subject, 
               const char* relation, 
               const char* object);

/* Legacy compatibility wrapper for simple fact addition */
void galileo_add_fact(GalileoModel* model, 
                     const char* subject, 
                     const char* relation, 
                     const char* object, 
                     float confidence);

/* =============================================================================
 * SYMBOLIC REASONING ENGINES
 * =============================================================================
 */

/* Main symbolic inference engine with safety and depth limiting */
void galileo_enhanced_symbolic_inference_safe(GalileoModel* model);

/* Legacy compatibility wrapper */
void galileo_symbolic_inference(GalileoModel* model);

/* =============================================================================
 * REASONING PATTERN IMPLEMENTATIONS
 * 
 * These functions implement specific logical reasoning patterns:
 * - Transitivity: A→B, B→C ⟹ A→C  
 * - Inheritance: All X are Y, Z is X ⟹ Z is Y (with exceptions)
 * - Quantified: "Most X are Y" ⟹ probabilistic facts
 * - Causal: A causes B causes C ⟹ A influences C
 * - Arithmetic: Basic numerical reasoning
 * =============================================================================
 */

/* Note: These are internal functions exposed for testing/debugging.
 * Normal usage should go through galileo_enhanced_symbolic_inference_safe()
 */

/* Pattern 1: Transitivity reasoning (A→B, B→C ⟹ A→C) */
/* Internal function - not exported in normal builds */
#ifdef GALILEO_EXPOSE_INTERNAL_REASONING
void apply_transitivity_reasoning(GalileoModel* model);
#endif

/* Pattern 2: Subclass inheritance with exceptions */
#ifdef GALILEO_EXPOSE_INTERNAL_REASONING
void apply_inheritance_reasoning(GalileoModel* model);
#endif

/* Pattern 3: Quantified reasoning ("most", "some", "all") */
#ifdef GALILEO_EXPOSE_INTERNAL_REASONING
void apply_quantified_reasoning(GalileoModel* model);
#endif

/* Pattern 4: Causal chain reasoning */
#ifdef GALILEO_EXPOSE_INTERNAL_REASONING
void apply_causal_reasoning(GalileoModel* model);
#endif

/* Pattern 5: Simple arithmetic reasoning */
#ifdef GALILEO_EXPOSE_INTERNAL_REASONING
void apply_arithmetic_reasoning(GalileoModel* model);
#endif

/* =============================================================================
 * CONFLICT DETECTION AND RESOLUTION
 * =============================================================================
 */

/* Conflict detection and resolution engine */
#ifdef GALILEO_EXPOSE_INTERNAL_REASONING
void detect_and_resolve_conflicts(GalileoModel* model);
#endif

/* =============================================================================
 * UTILITY FUNCTIONS
 * =============================================================================
 */

/* String normalization and comparison utilities */
#ifdef GALILEO_EXPOSE_INTERNAL_REASONING
void normalize_string(const char* input, char* output, size_t max_len);
int strings_equivalent(const char* str1, const char* str2);
float extract_numeric_value(const char* str);
#endif

/* =============================================================================
 * CONFIGURATION CONSTANTS
 * All constants can be overridden at compile time
 * =============================================================================
 */

/* Maximum reasoning depth to prevent infinite recursion */
#ifndef MAX_REASONING_DEPTH
#define MAX_REASONING_DEPTH 10
#endif

/* Maximum recursive inference iterations */
#ifndef MAX_RECURSIVE_INFERENCE
#define MAX_RECURSIVE_INFERENCE 5
#endif

/* Confidence thresholds for reasoning patterns */
#ifndef MIN_TRANSITIVITY_CONFIDENCE
#define MIN_TRANSITIVITY_CONFIDENCE 0.3f
#endif

#ifndef MIN_INHERITANCE_CONFIDENCE
#define MIN_INHERITANCE_CONFIDENCE 0.4f
#endif

#ifndef MIN_CAUSAL_CONFIDENCE
#define MIN_CAUSAL_CONFIDENCE 0.5f
#endif

/* Confidence decay factors for derived facts */
#ifndef TRANSITIVITY_DECAY_FACTOR
#define TRANSITIVITY_DECAY_FACTOR 0.8f
#endif

#ifndef INHERITANCE_DECAY_FACTOR
#define INHERITANCE_DECAY_FACTOR 0.9f
#endif

#ifndef QUANTIFIED_CONFIDENCE_FACTOR
#define QUANTIFIED_CONFIDENCE_FACTOR 0.7f
#endif

#ifndef CAUSAL_CHAIN_DECAY_FACTOR
#define CAUSAL_CHAIN_DECAY_FACTOR 0.6f
#endif

/* Conflict resolution parameters */
#ifndef EXCEPTION_SPECIFICITY_BOOST
#define EXCEPTION_SPECIFICITY_BOOST 1.2f
#endif

#ifndef CONFLICT_RESOLUTION_DECAY
#define CONFLICT_RESOLUTION_DECAY 0.3f
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
} SymbolicModuleInfo;

extern SymbolicModuleInfo symbolic_module_info;

#endif /* GALILEO_SYMBOLIC_H */
