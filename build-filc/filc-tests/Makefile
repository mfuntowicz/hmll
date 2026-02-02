# Makefile for fil-c hmll test suite
#
# Builds and runs functional tests for hmll compiled with Fil-C

FILC_ROOT ?= /home/ubuntu/filc-0.677-linux-x86_64
FILC_CC = $(FILC_ROOT)/build/bin/clang

# hmll paths
HMLL_ROOT = /home/ubuntu/hmll
HMLL_INCLUDE = $(HMLL_ROOT)/include
HMLL_BUILD = $(HMLL_ROOT)/build-filc
HMLL_LIB = $(HMLL_BUILD)/libhmll.a
FMT_LIB = $(HMLL_BUILD)/_deps/fmt-build/libfmt.a

# Compiler flags
CFLAGS = -I$(HMLL_INCLUDE) -O2 -g -Wall -Wextra
LDFLAGS = $(HMLL_LIB) $(FMT_LIB) -lstdc++ -lm

# Test programs
TESTS = test_source test_memory test_loader test_fetchv test_context test_mmap_view

.PHONY: all clean test help

all: $(TESTS)

# Build rule for all tests
%: %.c $(HMLL_LIB)
	@echo "Building $@ with Fil-C..."
	$(FILC_CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Run all tests
test: all
	@echo ""
	@echo "========================================"
	@echo "Running hmll Fil-C Test Suite"
	@echo "========================================"
	@echo ""
	@passed=0; failed=0; \
	for t in $(TESTS); do \
		echo "----------------------------------------"; \
		echo "Running: $$t"; \
		echo "----------------------------------------"; \
		if ./$$t; then \
			echo ""; \
			passed=$$((passed + 1)); \
		else \
			echo ""; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo "========================================"; \
	echo "Summary: $$passed passed, $$failed failed"; \
	echo "========================================"

# Run individual tests
test-source: test_source
	./test_source

test-memory: test_memory
	./test_memory

test-loader: test_loader
	./test_loader

test-fetchv: test_fetchv
	./test_fetchv

test-context: test_context
	./test_context

test-mmap: test_mmap_view
	./test_mmap_view

clean:
	rm -f $(TESTS) *.bin

help:
	@echo "hmll Fil-C Test Suite"
	@echo ""
	@echo "Usage:"
	@echo "  make          - Build all tests"
	@echo "  make test     - Build and run all tests"
	@echo "  make clean    - Remove built files"
	@echo ""
	@echo "Individual tests:"
	@echo "  make test-source  - Test source open/close"
	@echo "  make test-memory  - Test memory allocation"
	@echo "  make test-loader  - Test loader and fetch"
	@echo "  make test-fetchv  - Test vectorized fetch"
	@echo "  make test-context - Test context operations"
	@echo "  make test-mmap    - Test mmap views"
	@echo ""
	@echo "Environment:"
	@echo "  FILC_ROOT=/path/to/fil-c  (default: /home/ubuntu/filc-0.677-linux-x86_64)"
