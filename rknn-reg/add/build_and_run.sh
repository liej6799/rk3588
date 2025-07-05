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
LD_LIBRARY_PATH="/root/.pyenv/versions/3.11.4/lib" LD_PRELOAD="/data/Dev/rk3588/rknn-sniff/preload_python.so" ./add "$@" 