# =============================================================================
# Galileo v42 - Modular Build System
# Builds hot-swappable shared library modules + CLI executable
# ALL build artifacts go under ./build/ for proper build hygiene
# =============================================================================

# Project configuration
PROJECT_NAME = galileo
VERSION = 42
PREFIX = /usr/local

# Configurable build directory
BUILD_DIR ?= build

# Source directories (read-only)
SRC_DIR = src
INCLUDE_DIR = include
TEST_DIR = tests

# Build output directories (ALL under BUILD_DIR)
BUILD_LIB_DIR = $(BUILD_DIR)/lib/galileo
BUILD_BIN_DIR = $(BUILD_DIR)/bin
BUILD_INCLUDE_DIR = $(BUILD_DIR)/include
OBJ_DIR = $(BUILD_DIR)/obj
DEP_DIR = $(BUILD_DIR)/deps

# Create ALL necessary directories including the galileo subdirectory
$(shell mkdir -p $(OBJ_DIR)/core $(OBJ_DIR)/graph $(OBJ_DIR)/symbolic $(OBJ_DIR)/memory $(OBJ_DIR)/utils $(OBJ_DIR)/heuristic $(OBJ_DIR)/main $(OBJ_DIR)/tests)
$(shell mkdir -p $(DEP_DIR)/core $(DEP_DIR)/graph $(DEP_DIR)/symbolic $(DEP_DIR)/memory $(DEP_DIR)/utils $(DEP_DIR)/heuristic $(DEP_DIR)/main $(DEP_DIR)/tests)
$(shell mkdir -p $(BUILD_LIB_DIR) $(BUILD_BIN_DIR) $(BUILD_INCLUDE_DIR))

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c17 -fPIC -O3 -march=native -ffast-math
CFLAGS_DEBUG = -Wall -Wextra -std=c17 -fPIC -g -O0 -DDEBUG
LDFLAGS = -ldl -lm -pthread
SHARED_FLAGS = -shared -fPIC

# Include paths - use both source and build include dirs
INCLUDES = -I$(INCLUDE_DIR) -I$(BUILD_INCLUDE_DIR) -I$(SRC_DIR)

# Add the module loader to core sources
CORE_SOURCES = $(SRC_DIR)/core/galileo_core.c $(SRC_DIR)/core/galileo_module_loader.c
GRAPH_SOURCES = $(SRC_DIR)/graph/galileo_graph.c
SYMBOLIC_SOURCES = $(SRC_DIR)/symbolic/galileo_symbolic.c
MEMORY_SOURCES = $(SRC_DIR)/memory/galileo_memory.c
UTILS_SOURCES = $(SRC_DIR)/utils/galileo_utils.c
HEURISTIC_SOURCES = $(SRC_DIR)/heuristic/galileo_heuristic_compiler.c
MAIN_SOURCES = $(SRC_DIR)/main/galileo_main.c

# Test sources
TEST_SOURCES = $(TEST_DIR)/test_core.c \
               $(TEST_DIR)/test_graph.c \
               $(TEST_DIR)/test_symbolic.c \
               $(TEST_DIR)/test_memory.c \
               $(TEST_DIR)/test_integration.c \
               $(TEST_DIR)/test_runner.c

# Object files
CORE_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(CORE_SOURCES))
GRAPH_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(GRAPH_SOURCES))
SYMBOLIC_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SYMBOLIC_SOURCES))
MEMORY_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(MEMORY_SOURCES))
UTILS_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(UTILS_SOURCES))
HEURISTIC_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(HEURISTIC_SOURCES))
MAIN_OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(MAIN_SOURCES))
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.c,$(OBJ_DIR)/tests/%.o,$(TEST_SOURCES))

# Shared libraries (ALL under build/lib/galileo/ directory now)
CORE_LIB = $(BUILD_LIB_DIR)/libgalileo_core.so
GRAPH_LIB = $(BUILD_LIB_DIR)/libgalileo_graph.so
SYMBOLIC_LIB = $(BUILD_LIB_DIR)/libgalileo_symbolic.so
MEMORY_LIB = $(BUILD_LIB_DIR)/libgalileo_memory.so
UTILS_LIB = $(BUILD_LIB_DIR)/libgalileo_utils.so
HEURISTIC_LIB = $(BUILD_LIB_DIR)/libgalileo_heuristic.so

# Executables (ALL under build directory)
MAIN_EXECUTABLE = $(BUILD_BIN_DIR)/galileo
TEST_EXECUTABLE = $(BUILD_BIN_DIR)/galileo_test

# Dependency files
CORE_DEPS = $(patsubst $(SRC_DIR)/%.c,$(DEP_DIR)/%.d,$(CORE_SOURCES))
GRAPH_DEPS = $(patsubst $(SRC_DIR)/%.c,$(DEP_DIR)/%.d,$(GRAPH_SOURCES))
SYMBOLIC_DEPS = $(patsubst $(SRC_DIR)/%.c,$(DEP_DIR)/%.d,$(SYMBOLIC_SOURCES))
MEMORY_DEPS = $(patsubst $(SRC_DIR)/%.c,$(DEP_DIR)/%.d,$(MEMORY_SOURCES))
UTILS_DEPS = $(patsubst $(SRC_DIR)/%.c,$(DEP_DIR)/%.d,$(UTILS_SOURCES))
HEURISTIC_DEPS = $(patsubst $(SRC_DIR)/%.c,$(DEP_DIR)/%.d,$(HEURISTIC_SOURCES))
MAIN_DEPS = $(patsubst $(SRC_DIR)/%.c,$(DEP_DIR)/%.d,$(MAIN_SOURCES))
TEST_DEPS = $(patsubst $(TEST_DIR)/%.c,$(DEP_DIR)/tests/%.d,$(TEST_SOURCES))

ALL_DEPS = $(CORE_DEPS) $(GRAPH_DEPS) $(SYMBOLIC_DEPS) $(MEMORY_DEPS) $(UTILS_DEPS) $(HEURISTIC_DEPS) $(MAIN_DEPS) $(TEST_DEPS)

# Color output for better visibility
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
BLUE = \033[0;34m
PURPLE = \033[0;35m
CYAN = \033[0;36m
NC = \033[0m # No Color

# Default target
.PHONY: all
all: banner copy-headers libs executable
	@echo -e "$(GREEN)✅ Build complete! Galileo v$(VERSION) ready to rock!$(NC)"
	@echo -e "$(BLUE)Run: ./$(MAIN_EXECUTABLE) --help$(NC)"

.PHONY: banner
banner:
	@echo -e "$(YELLOW)"
	@echo -e "  ╔═══════════════════════════════════════════════╗"
	@echo -e "  ║          🚀 Building Galileo v$(VERSION) 🚀          ║"
	@echo -e "  ║    Graph-and-Logic Integrated Language       ║"
	@echo -e "  ║           Engine (Modular Edition)           ║"
	@echo -e "  ╚═══════════════════════════════════════════════╝"
	@echo -e "$(NC)"

# Copy headers to build directory for clean include structure
.PHONY: copy-headers
copy-headers:
	@echo -e "$(BLUE)📄 Copying headers to build directory...$(NC)"
	@cp -r $(INCLUDE_DIR)/* $(BUILD_INCLUDE_DIR)/ 2>/dev/null || echo -e "$(YELLOW)No headers in $(INCLUDE_DIR) to copy$(NC)"
	@mkdir -p $(BUILD_INCLUDE_DIR)/galileo
	@cp $(SRC_DIR)/*/*.h $(BUILD_INCLUDE_DIR)/galileo/ 2>/dev/null || echo -e "$(YELLOW)No module headers to copy$(NC)"
	@echo -e "$(GREEN)✅ Headers copied to $(BUILD_INCLUDE_DIR)$(NC)"

# Build all shared libraries
.PHONY: libs
libs: copy-headers $(CORE_LIB) $(GRAPH_LIB) $(SYMBOLIC_LIB) $(MEMORY_LIB) $(UTILS_LIB) $(HEURISTIC_LIB)
	@echo -e "$(GREEN)📚 All shared libraries built successfully!$(NC)"
	@echo -e "$(CYAN)Libraries installed to: $(BUILD_LIB_DIR)$(NC)"

# Build main executable - NO DEPENDENCY ON LIBS!
.PHONY: executable
executable: copy-headers $(MAIN_EXECUTABLE)
	@echo -e "$(GREEN)🎯 Dynamic executable ready! No static module dependencies.$(NC)"

# Core module shared library
$(CORE_LIB): $(CORE_OBJECTS)
	@echo -e "$(BLUE)🔧 Building core module...$(NC)"
	$(CC) $(SHARED_FLAGS) -o $@ $^ $(LDFLAGS)
	@echo -e "$(GREEN)✅ Core module: $@$(NC)"

# Graph module shared library  
$(GRAPH_LIB): $(GRAPH_OBJECTS) $(CORE_LIB)
	@echo -e "$(BLUE)🔧 Building graph module...$(NC)"
	$(CC) $(SHARED_FLAGS) -o $@ $(GRAPH_OBJECTS) -L$(BUILD_LIB_DIR) -lgalileo_core $(LDFLAGS)
	@echo -e "$(GREEN)✅ Graph module: $@$(NC)"

# Symbolic module shared library
$(SYMBOLIC_LIB): $(SYMBOLIC_OBJECTS) $(CORE_LIB)
	@echo -e "$(BLUE)🔧 Building symbolic module...$(NC)"
	$(CC) $(SHARED_FLAGS) -o $@ $(SYMBOLIC_OBJECTS) -L$(BUILD_LIB_DIR) -lgalileo_core $(LDFLAGS)
	@echo -e "$(GREEN)✅ Symbolic module: $@$(NC)"

# Memory module shared library
$(MEMORY_LIB): $(MEMORY_OBJECTS) $(CORE_LIB)
	@echo -e "$(BLUE)🔧 Building memory module...$(NC)"
	$(CC) $(SHARED_FLAGS) -o $@ $(MEMORY_OBJECTS) -L$(BUILD_LIB_DIR) -lgalileo_core $(LDFLAGS)
	@echo -e "$(GREEN)✅ Memory module: $@$(NC)"

# Utils module shared library
$(UTILS_LIB): $(UTILS_OBJECTS) $(CORE_LIB)
	@echo -e "$(BLUE)🔧 Building utils module...$(NC)"
	$(CC) $(SHARED_FLAGS) -o $@ $(UTILS_OBJECTS) -L$(BUILD_LIB_DIR) -lgalileo_core $(LDFLAGS)
	@echo -e "$(GREEN)✅ Utils module: $@$(NC)"

# Heuristic module shared library (NEW!)
$(HEURISTIC_LIB): $(HEURISTIC_OBJECTS) $(CORE_LIB)
	@echo -e "$(BLUE)🔧 Building heuristic compiler module...$(NC)"
	$(CC) $(SHARED_FLAGS) -o $@ $(HEURISTIC_OBJECTS) -L$(BUILD_LIB_DIR) -lgalileo_core $(LDFLAGS) -lsqlite3
	@echo -e "$(GREEN)✅ Heuristic module: $@ (with SQLite3 support)$(NC)"

# Main executable (dynamic loading with core module for bootstrapping)
$(MAIN_EXECUTABLE): $(MAIN_OBJECTS) $(CORE_LIB)
	@echo -e "$(BLUE)🔧 Building main executable (dynamic loading with core bootstrap)...$(NC)"
	$(CC) -o $@ $(MAIN_OBJECTS) -L$(BUILD_LIB_DIR) -lgalileo_core $(LDFLAGS) -Wl,-rpath,$(shell pwd)/$(BUILD_LIB_DIR)
	@echo -e "$(GREEN)✅ Main executable: $@ (with dynamic core bootstrap!)$(NC)"

# Test executable (still links statically for testing purposes)
$(TEST_EXECUTABLE): $(TEST_OBJECTS) libs
	@echo -e "$(BLUE)🔧 Building test suite (with static module linking for testing)...$(NC)"
	$(CC) -o $@ $(TEST_OBJECTS) -L$(BUILD_LIB_DIR) -lgalileo_core -lgalileo_graph -lgalileo_symbolic -lgalileo_memory -lgalileo_utils -lgalileo_heuristic $(LDFLAGS) -lsqlite3 -Wl,-rpath,$(shell pwd)/$(BUILD_LIB_DIR)
	@echo -e "$(GREEN)✅ Test executable: $@ (with static linking for comprehensive testing)$(NC)"

# Object file compilation with dependency generation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@echo -e "$(YELLOW)Compiling $<...$(NC)"
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -MF $(patsubst $(OBJ_DIR)/%.o,$(DEP_DIR)/%.d,$@) -c $< -o $@

$(OBJ_DIR)/tests/%.o: $(TEST_DIR)/%.c
	@echo -e "$(YELLOW)Compiling test $<...$(NC)"
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -MMD -MP -MF $(patsubst $(OBJ_DIR)/tests/%.o,$(DEP_DIR)/tests/%.d,$@) -c $< -o $@

# Include dependency files
-include $(ALL_DEPS)

# Create convenience symlinks in root (optional)
.PHONY: symlinks
symlinks: all
	@echo -e "$(BLUE)🔗 Creating convenience symlinks...$(NC)"
	@ln -sf $(MAIN_EXECUTABLE) galileo
	@echo -e "$(GREEN)✅ Created symlink: galileo -> $(MAIN_EXECUTABLE)$(NC)"

# Debug build
.PHONY: debug
debug: CFLAGS = $(CFLAGS_DEBUG)
debug: clean all
	@echo -e "$(GREEN)🐛 Debug build complete!$(NC)"

# Release build (optimized)
.PHONY: release
release: CFLAGS += -DNDEBUG -flto
release: clean all symlinks
	@echo -e "$(GREEN)🚀 Release build complete!$(NC)"

# Run tests (static linking for comprehensive testing)
.PHONY: test
test: $(TEST_EXECUTABLE)
	@echo -e "$(BLUE)🧪 Running test suite (with static module linking)...$(NC)"
	@export LD_LIBRARY_PATH=$(shell pwd)/$(BUILD_LIB_DIR):$LD_LIBRARY_PATH && ./$(TEST_EXECUTABLE)

# Run integration tests with main executable (no LD_LIBRARY_PATH!)
.PHONY: test-integration  
test-integration: all
	@echo -e "$(BLUE)🔄 Running dynamic loading integration tests...$(NC)"
	@./$(MAIN_EXECUTABLE) --test

# Quick demo (no LD_LIBRARY_PATH needed!)
.PHONY: demo
demo: all
	@echo -e "$(BLUE)🎭 Running dynamic loading demo...$(NC)"
	@echo -e "All men are mortal. Socrates is a man. Is Socrates mortal?" | ./$(MAIN_EXECUTABLE) --verbose

# Interactive mode (no LD_LIBRARY_PATH needed!)
.PHONY: interactive
interactive: all
	@echo -e "$(BLUE)💬 Starting dynamic interactive mode...$(NC)"
	@./$(MAIN_EXECUTABLE) --interactive

# Performance test with long input (no LD_LIBRARY_PATH!)
.PHONY: perf-test
perf-test: all
	@echo -e "$(BLUE)⚡ Running dynamic loading performance test...$(NC)"
	@seq 1 1000 | xargs -I {} echo -e "Token{} is related to concept." | \
	./$(MAIN_EXECUTABLE) --verbose --max-iterations 10

# Install to system (updated for galileo subdirectory)
.PHONY: install
install: all
	@echo -e "$(BLUE)📦 Installing Galileo v$(VERSION)...$(NC)"
	install -d $(PREFIX)/bin
	install -d $(PREFIX)/lib/galileo
	install -d $(PREFIX)/include/galileo
	install -m 755 $(MAIN_EXECUTABLE) $(PREFIX)/bin/
	install -m 755 $(BUILD_LIB_DIR)/*.so $(PREFIX)/lib/galileo/
	install -m 644 $(BUILD_INCLUDE_DIR)/*.h $(PREFIX)/include/galileo/ 2>/dev/null || echo -e "No main headers to install"
	install -m 644 $(BUILD_INCLUDE_DIR)/galileo/*.h $(PREFIX)/include/galileo/ 2>/dev/null || echo -e "No module headers to install"
	ldconfig
	@echo -e "$(GREEN)✅ Installed to $(PREFIX) (libraries in lib/galileo/)$(NC)"
	@echo -e "$(YELLOW)📝 Note: SQLite3 development packages required for heuristic module$(NC)"

# Uninstall from system (updated for galileo subdirectory)
.PHONY: uninstall
uninstall:
	@echo -e "$(BLUE)🗑️  Uninstalling Galileo...$(NC)"
	rm -f $(PREFIX)/bin/galileo
	rm -rf $(PREFIX)/lib/galileo
	rm -rf $(PREFIX)/include/galileo
	ldconfig
	@echo -e "$(GREEN)✅ Uninstalled$(NC)"

# Package for distribution (updated for galileo subdirectory)
.PHONY: package
package: clean release
	@echo -e "$(BLUE)📦 Creating distribution package...$(NC)"
	mkdir -p dist/galileo-v$(VERSION)
	cp -r $(SRC_DIR) $(INCLUDE_DIR) $(TEST_DIR) Makefile README.md dist/galileo-v$(VERSION)/ 2>/dev/null || echo -e "Some files not found, continuing..."
	cp $(BUILD_BIN_DIR)/galileo dist/galileo-v$(VERSION)/ 2>/dev/null || echo -e "Binary not found, continuing..."
	mkdir -p dist/galileo-v$(VERSION)/lib/galileo
	cp $(BUILD_LIB_DIR)/*.so dist/galileo-v$(VERSION)/lib/galileo/ 2>/dev/null || echo -e "Libraries not found, continuing..."
	cd dist && tar czf galileo-v$(VERSION).tar.gz galileo-v$(VERSION)
	@echo -e "$(GREEN)✅ Package created: dist/galileo-v$(VERSION).tar.gz$(NC)"

# Code analysis
.PHONY: lint
lint:
	@echo -e "$(BLUE)🔍 Running code analysis...$(NC)"
	@cppcheck --enable=all --std=c17 $(SRC_DIR)/ $(TEST_DIR)/ 2>/dev/null || echo -e "$(YELLOW)cppcheck not found, skipping...$(NC)"
	@clang-tidy $(SRC_DIR)/*/*.c -- $(INCLUDES) 2>/dev/null || echo -e "$(YELLOW)clang-tidy not found, skipping...$(NC)"

# Memory leak check (tests still need LD_LIBRARY_PATH for static linking)
.PHONY: valgrind
valgrind: debug
	@echo -e "$(BLUE)🔍 Running valgrind memory check on dynamic executable...$(NC)"
	@echo -e "Test input" | valgrind --leak-check=full --show-leak-kinds=all ./$(MAIN_EXECUTABLE) || echo -e "$(YELLOW)valgrind not found or test failed$(NC)"

# Static analysis with multiple tools
.PHONY: analyze
analyze: lint
	@echo -e "$(BLUE)🔬 Running comprehensive static analysis...$(NC)"
	@scan-build make clean all 2>/dev/null || echo -e "$(YELLOW)scan-build not found, skipping...$(NC)"
	@pvs-studio-analyzer trace -- make clean all 2>/dev/null || echo -e "$(YELLOW)PVS-Studio not found, skipping...$(NC)"

# Cleanup - ONLY remove build directory
.PHONY: clean
clean:
	@echo -e "$(BLUE)🧹 Cleaning build artifacts...$(NC)"
	rm -rf $(BUILD_DIR)/
	rm -f galileo  # Remove convenience symlink
	rm -f galileo_knowledge.db  # Remove SQLite database if exists
	@echo -e "$(GREEN)✅ Clean complete!$(NC)"

# Deep clean (including dist)
.PHONY: distclean
distclean: clean
	@echo -e "$(BLUE)🧹 Deep cleaning...$(NC)"
	rm -rf dist/
	@echo -e "$(GREEN)✅ Deep clean complete!$(NC)"

# Show module loading capabilities (no LD_LIBRARY_PATH needed for discovery!)
.PHONY: modules
modules: $(MAIN_EXECUTABLE)
	@echo -e "$(BLUE)📚 Checking module loading capabilities...$(NC)"
	@./$(MAIN_EXECUTABLE) --list-modules

# Show build status
.PHONY: status
status:
	@echo -e "$(PURPLE)📊 Build Status:$(NC)"
	@echo -e "$(CYAN)Libraries in $(BUILD_LIB_DIR):$(NC)"
	@for lib in $(CORE_LIB) $(GRAPH_LIB) $(SYMBOLIC_LIB) $(MEMORY_LIB) $(UTILS_LIB) $(HEURISTIC_LIB); do \
		if [ -f "$$lib" ]; then \
			echo -e "  ✅ $$lib"; \
		else \
			echo -e "  ❌ $$lib (missing)"; \
		fi; \
	done
	@echo -e "$(CYAN)Executables:$(NC)"
	@for exe in $(MAIN_EXECUTABLE) $(TEST_EXECUTABLE); do \
		if [ -f "$$exe" ]; then \
			echo -e "  ✅ $$exe"; \
		else \
			echo -e "  ❌ $$exe (missing)"; \
		fi; \
	done
	@echo -e "$(CYAN)Build directory structure:$(NC)"
	@ls -la $(BUILD_DIR)/ 2>/dev/null || echo -e "  No build directory found"
	@echo -e "$(CYAN)Module directory contents:$(NC)"
	@ls -la $(BUILD_LIB_DIR)/ 2>/dev/null || echo -e "  No module directory found"

# Show file structure
.PHONY: tree
tree:
	@echo -e "$(BLUE)📁 Project structure:$(NC)"
	@find . -name "*.c" -o -name "*.h" -o -name "Makefile" -o -name "*.md" | grep -v $(BUILD_DIR) | sort

# Show build directory contents
.PHONY: build-tree
build-tree:
	@echo -e "$(BLUE)📁 Build directory structure:$(NC)"
	@find $(BUILD_DIR) -type f 2>/dev/null | sort || echo -e "  No build directory found"

# Dependency check - ensure SQLite3 is available
.PHONY: check-deps
check-deps:
	@echo -e "$(BLUE)🔍 Checking dependencies...$(NC)"
	@which sqlite3 >/dev/null || echo -e "$(YELLOW)⚠️  SQLite3 not found in PATH$(NC)"
	@pkg-config --exists sqlite3 2>/dev/null && echo -e "$(GREEN)✅ SQLite3 development libraries found$(NC)" || echo -e "$(YELLOW)⚠️  SQLite3 development libraries not found$(NC)"
	@echo -e "$(CYAN)Install SQLite3 development packages:$(NC)"
	@echo -e "  Ubuntu/Debian: sudo apt-get install libsqlite3-dev"
	@echo -e "  CentOS/RHEL:   sudo yum install sqlite-devel"
	@echo -e "  macOS:         brew install sqlite3"

# Help target
.PHONY: help
help:
	@echo -e "$(YELLOW)🚀 Galileo v$(VERSION) Build System Help$(NC)"
	@echo -e ""
	@echo -e "$(BLUE)Primary Targets:$(NC)"
	@echo -e "  all              - Build everything (default)"
	@echo -e "  libs             - Build shared library modules"
	@echo -e "  executable       - Build main executable"
	@echo -e "  test             - Build and run test suite"
	@echo -e "  clean            - Remove build artifacts"
	@echo -e ""
	@echo -e "$(BLUE)Build Variants:$(NC)"
	@echo -e "  debug            - Build with debug symbols"
	@echo -e "  release          - Build optimized release"
	@echo -e "  package          - Create distribution package"
	@echo -e ""
	@echo -e "$(BLUE)Dependencies:$(NC)"
	@echo -e "  check-deps       - Check for required dependencies"
	@echo -e "  libsqlite3-dev   - Required for heuristic compiler module"
	@echo -e ""
	@echo -e "$(BLUE)Convenience:$(NC)"
	@echo -e "  symlinks         - Create convenience symlinks in project root"
	@echo -e "  copy-headers     - Copy headers to build directory"
	@echo -e "  build-tree       - Show build directory structure"
	@echo -e ""
	@echo -e "$(BLUE)Testing & Analysis:$(NC)"
	@echo -e "  test             - Run unit tests"
	@echo -e "  test-integration - Run integration tests"
	@echo -e "  demo             - Quick demo run"
	@echo -e "  interactive      - Start interactive mode"
	@echo -e "  perf-test        - Performance test"
	@echo -e "  lint             - Code analysis"
	@echo -e "  valgrind         - Memory leak check"
	@echo -e "  analyze          - Comprehensive static analysis"
	@echo -e ""
	@echo -e "$(BLUE)Installation:$(NC)"
	@echo -e "  install          - Install to system ($(PREFIX))"
	@echo -e "  uninstall        - Remove from system"
	@echo -e ""
	@echo -e "$(BLUE)Development:$(NC)"
	@echo -e "  modules          - Show module capabilities"
	@echo -e "  tree             - Show project structure"
	@echo -e "  status           - Show build status"
	@echo -e "  distclean        - Deep clean including dist/"
	@echo -e ""
	@echo -e "$(BLUE)Build Configuration:$(NC)"
	@echo -e "  BUILD_DIR        - Build output directory (default: build)"
	@echo -e ""
	@echo -e "$(BLUE)Module Information:$(NC)"
	@echo -e "  Shared libraries are now built in: $(BUILD_LIB_DIR)"
	@echo -e "  This enables proper module discovery and hot-loading"
	@echo -e "  Heuristic module provides GA-derived fact extraction with SQLite caching"
	@echo -e ""
	@echo -e "$(BLUE)Examples:$(NC)"
	@echo -e "  make check-deps           # Check dependencies"
	@echo -e "  make debug test           # Debug build and test"
	@echo -e "  make release install      # Release build and install"
	@echo -e "  make demo                 # Quick demo"
	@echo -e "  make status               # Check what's built"
	@echo -e "  BUILD_DIR=mybuild make    # Custom build directory"
