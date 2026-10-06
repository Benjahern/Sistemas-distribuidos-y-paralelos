# Compila cada módulo por separado: no enlaza ni exige que otros módulos
# o main() estén implementados todavía.

CXX       ?= g++
CPPFLAGS  ?= -Iinclude
CXXFLAGS  ?= -std=c++17 -Wall -Wextra -Wpedantic -O2
OPENMP    ?= -fopenmp
BUILD_DIR ?= build

# Direcciones para encontrar los archivos correspondientes
SOURCES   := $(sort $(wildcard src/*.cpp))
MODULES   := $(basename $(notdir $(SOURCES)))
OBJECTS   := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
TESTS     := $(sort $(wildcard tests/*.cpp))
TEST_OBJS := $(patsubst tests/%.cpp,$(BUILD_DIR)/tests/%.o,$(TESTS))

.DEFAULT_GOAL := all
.PHONY: all clean test test-role2 test-role3 test-integration $(MODULES)

# make: compila todos los .cpp disponibles como unidades independientes.
all: $(OBJECTS)

# También permite compilar solo una parte, por ejemplo: make Butterfly1D
$(MODULES): %: $(BUILD_DIR)/%.o

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -MMD -MP -c $< -o $@

test: test-role2 test-role3

test-role2: $(BUILD_DIR)/test_butterfly1d
	$(abspath $(BUILD_DIR)/test_butterfly1d)

$(BUILD_DIR)/test_butterfly1d: $(BUILD_DIR)/tests/test_butterfly1d.o $(BUILD_DIR)/Butterfly1D.o
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

test-role3: $(BUILD_DIR)/test_fft2d
	$(abspath $(BUILD_DIR)/test_fft2d)

$(BUILD_DIR)/test_fft2d: $(BUILD_DIR)/tests/standalone/test_fft2d.o $(BUILD_DIR)/Fft2D.o
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

$(BUILD_DIR)/tests/standalone/test_fft2d.o: tests/test_fft2d.cpp
	@mkdir -p $(BUILD_DIR)/tests/standalone
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -DFFT2D_STANDALONE_STUBS -MMD -MP -c $< -o $@

test-integration: $(BUILD_DIR)/test_fft2d_integration
	$(abspath $(BUILD_DIR)/test_fft2d_integration)

$(BUILD_DIR)/test_fft2d_integration: $(BUILD_DIR)/tests/test_fft2d.o $(BUILD_DIR)/Fft2D.o $(BUILD_DIR)/ComplexField.o $(BUILD_DIR)/Butterfly1D.o
	$(CXX) $(CXXFLAGS) $(OPENMP) $^ -o $@

$(BUILD_DIR)/tests/%.o: tests/%.cpp
	@mkdir -p $(BUILD_DIR)/tests
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -MMD -MP -c $< -o $@

# Elimina los archivos de la carpeta generada build
clean:
	rm -rf $(BUILD_DIR)

-include $(OBJECTS:.o=.d) $(TEST_OBJS:.o=.d) $(BUILD_DIR)/tests/standalone/test_fft2d.d
