CXX ?= g++
CXXFLAGS ?= -Wall -Wextra -O3 -std=c++17 -fopenmp
LDFLAGS ?= -fopenmp
CPPFLAGS ?= -Iinclude

TARGET := fft2d
SOURCES := $(wildcard src/*.cpp)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SOURCES) $(wildcard include/*.h)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SOURCES) -o $@ $(LDFLAGS)

test: all
	@echo "No hay pruebas configuradas en tests/." >&2
	@false

clean:
	rm -f $(TARGET)
