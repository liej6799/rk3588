# Simple Makefile for rknnops program
CC = gcc
CFLAGS = -Wall -Wextra -std=c99
TARGET = rknnops
SOURCE = rknnops.c

# Default target
all: $(TARGET)

# Build the rknnops program
$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE)

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run