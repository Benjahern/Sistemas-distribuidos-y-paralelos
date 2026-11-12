# Laboratorio 2: Programación GPGPU con CUDA (Filtrado de imagen 2D y Sobel)
# Soporta compilación con nvcc (CUDA completo) y fallback automático a g++ si nvcc no está instalado.

NVCC      ?= nvcc
CXX       ?= g++
PYTHON    ?= python3
BUILD_DIR ?= build
OUTPUT    ?= results
CPPFLAGS  ?= -Iinclude

HAS_NVCC  := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(HAS_NVCC),1)
    COMPILER   := $(NVCC)
    NVCCFLAGS  ?= -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -Wextra
    COMPILE_CPP = $(COMPILER) $(CPPFLAGS) $(NVCCFLAGS) -x cu -c $< -o $@
    COMPILE_CU  = $(COMPILER) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@
    LINK_CMD    = $(COMPILER) $(NVCCFLAGS) $^ -o $@
else
    COMPILER   := $(CXX)
    CXXFLAGS   ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
    COMPILE_CPP = $(COMPILER) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
    COMPILE_CU  = $(COMPILER) $(CPPFLAGS) $(CXXFLAGS) -x c++ -c $< -o $@
    LINK_CMD    = $(COMPILER) $(CXXFLAGS) $^ -o $@
endif

TARGET := $(BUILD_DIR)/image_cuda

CPP_SOURCES := $(sort $(wildcard src/*.cpp))
CU_SOURCES  := $(sort $(wildcard src/*.cu))
CPP_OBJECTS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CU_OBJECTS  := $(patsubst src/%.cu,$(BUILD_DIR)/%_cu.o,$(CU_SOURCES))
OBJECTS     := $(CPP_OBJECTS) $(CU_OBJECTS)
LIB_OBJS    := $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS))

.DEFAULT_GOAL := all
.PHONY: all clean test test-cpu demo plots

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(BUILD_DIR)
	$(LINK_CMD)

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(COMPILE_CPP)

$(BUILD_DIR)/%_cu.o: src/%.cu
	@mkdir -p $(BUILD_DIR)
	$(COMPILE_CU)

$(BUILD_DIR)/tests/%.o: tests/%.cpp
	@mkdir -p $(BUILD_DIR)/tests
	$(COMPILE_CPP)

$(BUILD_DIR)/test_image_buffer: $(BUILD_DIR)/tests/test_image_buffer.o $(BUILD_DIR)/ImageBuffer.o
	$(LINK_CMD)

$(BUILD_DIR)/test_pipeline_cpu_gpu: $(BUILD_DIR)/tests/test_pipeline_cpu_gpu.o $(LIB_OBJS)
	$(LINK_CMD)

$(BUILD_DIR)/test_cpu_reference: $(BUILD_DIR)/tests/test_cpu_reference.o $(BUILD_DIR)/ImageBuffer.o $(BUILD_DIR)/Convolution2D.o $(BUILD_DIR)/Sobel.o $(BUILD_DIR)/Convolution2D_cu.o $(BUILD_DIR)/Sobel_cu.o
	$(LINK_CMD)

test-cpu: $(BUILD_DIR)/test_image_buffer $(BUILD_DIR)/test_cpu_reference
	"$(abspath $(BUILD_DIR)/test_image_buffer)"
	"$(abspath $(BUILD_DIR)/test_cpu_reference)"

test: all test-cpu $(BUILD_DIR)/test_pipeline_cpu_gpu
	"$(abspath $(BUILD_DIR)/test_pipeline_cpu_gpu)"
	LAB2_EXECUTABLE="$(abspath $(TARGET))" LAB2_TEST_DIR="$(abspath $(BUILD_DIR))" $(PYTHON) tests/test_outputs.py

demo: all
	"$(abspath $(TARGET))" demo --output "$(OUTPUT)"

plots:
	$(PYTHON) scripts/plot_results.py --input "$(OUTPUT)"

clean:
	rm -rf "$(BUILD_DIR)"

