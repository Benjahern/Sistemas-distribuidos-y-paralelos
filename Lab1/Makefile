# Compila cada módulo por separado: no enlaza ni exige que otros módulos
# o main() estén implementados todavía.

CXX       ?= g++ # Compilador de c++
CPPFLAGS  ?= -Iinclude # esto es opciones del preprocesador para permitir encontrar las cabeceras
CXXFLAGS  ?= -std=c++17 -Wall -Wextra -Wpedantic -O2 # advertencias y para optimizacion
OPENMP    ?= -fopenmp # para activar el OpenMP
BUILD_DIR ?= build # directorio de construcción

# Direcciones para encontrar los archivos correspondientes
SOURCES   := $(sort $(wildcard src/*.cpp))
MODULES   := $(basename $(notdir $(SOURCES)))
OBJECTS   := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))
TESTS     := $(sort $(wildcard tests/*.cpp))
TEST_OBJS := $(patsubst tests/%.cpp,$(BUILD_DIR)/tests/%.o,$(TESTS))

.DEFAULT_GOAL := all
.PHONY: all clean test $(MODULES)

# make: compila todos los .cpp disponibles como unidades independientes.
all: $(OBJECTS)

# También permite compilar solo una parte, por ejemplo: make Butterfly1D
$(MODULES): %: $(BUILD_DIR)/%.o

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -MMD -MP -c $< -o $@

# Las pruebas también se compilan de forma independiente: así pueden existir
# referencias a módulos que otra rama todavía no haya implementado.
test: $(TEST_OBJS)
	@if [ -z "$(TESTS)" ]; then \
	  echo "No hay archivos de prueba en tests/*.cpp todavía."; \
	else \
	  echo "Pruebas compiladas. El enlace/ejecución depende de las pruebas de cada módulo."; \
	fi

$(BUILD_DIR)/tests/%.o: tests/%.cpp
	@mkdir -p $(BUILD_DIR)/tests
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP) -MMD -MP -c $< -o $@

# Elimina los archhivos de la carpeta generada build
clean:
	rm -rf $(BUILD_DIR)

-include $(OBJECTS:.o=.d) $(TEST_OBJS:.o=.d)
