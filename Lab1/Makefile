# Aplicación integrada y targets de módulos para desarrollo independiente.

CXX       ?= g++
CPPFLAGS  ?= -Iinclude
CXXFLAGS  ?= -std=c++17 -Wall -Wextra -Wpedantic -O2
OPENMP    ?= -fopenmp
BUILD_DIR ?= build
TEST_THREADS ?= 4
PYTHON    ?= python3
OUTPUT    ?= results

# Direcciones para encontrar los archivos correspondientes
SOURCES   := $(sort $(wildcard src/*.cpp))
MODULES   := $(basename $(notdir $(SOURCES)))
OBJECTS   := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
LIB_OBJS  := $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS))
TESTS     := $(sort $(wildcard tests/*.cpp))
TEST_OBJS := $(patsubst tests/%.cpp,$(BUILD_DIR)/tests/%.o,$(TESTS))

.DEFAULT_GOAL := all
.PHONY: all clean test test-role2 test-role3 test-integration \
        demo plots benchmark pipeline $(MODULES)

all: $(BUILD_DIR)/fft2d

$(BUILD_DIR)/fft2d: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

# También permite compilar solo una parte, por ejemplo: make Butterfly1D
$(MODULES): %: $(BUILD_DIR)/%.o

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -MMD -MP -c $< -o $@

test: all test-role2 test-role3 test-integration $(BUILD_DIR)/test_complexfield $(BUILD_DIR)/test_metrics_benchmark
	"$(abspath $(BUILD_DIR)/test_complexfield)"
	"$(abspath $(BUILD_DIR)/test_metrics_benchmark)"
	FFT_EXECUTABLE="$(abspath $(BUILD_DIR)/fft2d)" FFT_TEST_DIR="$(abspath $(BUILD_DIR))" $(PYTHON) tests/test_outputs.py

$(BUILD_DIR)/test_complexfield: $(BUILD_DIR)/tests/test_complexfield.o $(BUILD_DIR)/ComplexField.o
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

$(BUILD_DIR)/test_metrics_benchmark: $(BUILD_DIR)/tests/test_metrics_benchmark.o $(LIB_OBJS)
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

benchmark: all
	$(PYTHON) scripts/run_benchmark.py --executable "$(abspath $(BUILD_DIR)/fft2d)" --output "$(OUTPUT)" -- $(BENCHMARK_ARGS)

demo: all
	"$(abspath $(BUILD_DIR)/fft2d)" demo --output "$(OUTPUT)"

plots:
	$(PYTHON) scripts/plot_results.py --input "$(OUTPUT)"

# Ejecutar en orden, también con make -j: no medir mientras corren pruebas.
pipeline:
	$(MAKE) test
	$(MAKE) demo
	$(MAKE) benchmark
	$(MAKE) plots

test-role2: $(BUILD_DIR)/test_butterfly1d
	OMP_NUM_THREADS=$(TEST_THREADS) "$(abspath $(BUILD_DIR)/test_butterfly1d)"

$(BUILD_DIR)/test_butterfly1d: $(BUILD_DIR)/tests/test_butterfly1d.o $(BUILD_DIR)/Butterfly1D.o
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

test-role3: $(BUILD_DIR)/test_fft2d
	OMP_NUM_THREADS=$(TEST_THREADS) "$(abspath $(BUILD_DIR)/test_fft2d)"

$(BUILD_DIR)/test_fft2d: $(BUILD_DIR)/tests/standalone/test_fft2d.o $(BUILD_DIR)/Fft2D.o
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

$(BUILD_DIR)/tests/standalone/test_fft2d.o: tests/test_fft2d.cpp
	@mkdir -p $(BUILD_DIR)/tests/standalone
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -DFFT2D_STANDALONE_STUBS -MMD -MP -c $< -o $@

test-integration: $(BUILD_DIR)/test_fft2d_integration
	OMP_NUM_THREADS=$(TEST_THREADS) "$(abspath $(BUILD_DIR)/test_fft2d_integration)"

$(BUILD_DIR)/test_fft2d_integration: $(BUILD_DIR)/tests/test_fft2d.o $(BUILD_DIR)/Fft2D.o $(BUILD_DIR)/ComplexField.o $(BUILD_DIR)/Butterfly1D.o
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

$(BUILD_DIR)/tests/%.o: tests/%.cpp
	@mkdir -p $(BUILD_DIR)/tests
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -MMD -MP -c $< -o $@

# Elimina los archivos de la carpeta generada build
clean:
	rm -rf "$(BUILD_DIR)"

-include $(OBJECTS:.o=.d) $(TEST_OBJS:.o=.d) $(BUILD_DIR)/tests/standalone/test_fft2d.d
