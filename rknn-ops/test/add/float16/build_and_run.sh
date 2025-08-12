#!/usr/bin/env bash
set -e

echo "=== Float16 NPU Test Build and Run Script ==="

# Parse command line arguments
CLEAN_BUILD=false
RUN_PROGRAM=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --build-only)
            RUN_PROGRAM=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --clean      Clean before building"
            echo "  --build-only Build but don't run"
            echo "  --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we need to regenerate model (following project pattern)
if [ -f "../../../rknn-reg/add/add.py" ]; then
    echo "Regenerating model..."
    cd ../../../rknn-reg/add
    python3 add.py
    cd - > /dev/null
fi

echo "Building float16 NPU test..."

if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning previous build..."
    make clean
fi

# Build using Makefile
make

if [ "$RUN_PROGRAM" = true ]; then
    echo "Running ./main ..."
    echo "Note: This program requires access to /dev/dri/card1 (NPU device)"
    ./main
else
    echo "Build completed. Use './main' to run the program."
fi

echo "=== Build and run completed successfully! ==="
