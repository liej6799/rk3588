#!/usr/bin/env bash
set -e

# Always regenerate the model
python3 add.py

# Create and enter build directory
BUILD_DIR=build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure and build
cmake ..
make

# Run the executable with any passed arguments
echo "Running ./add ..."
./add "$@" 