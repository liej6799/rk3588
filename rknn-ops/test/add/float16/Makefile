# Makefile for float16 NPU test program
CXX = g++
CXXFLAGS = -Wall -Wextra
TARGET = main
SOURCE = main.c

# Include paths
INCLUDES = -I/data/Dev/rk3588/rknn-header -I/data/Dev/rk3588/rknn-ops

# Libraries
LIBS = -ldrm

# Default target
all: $(TARGET)

# Build the main program
$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCE) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Install target (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

.PHONY: all clean run install

