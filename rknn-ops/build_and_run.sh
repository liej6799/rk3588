#!/usr/bin/env bash
set -e

echo "=== RKNN-OPS Build and Run Script ==="

# Parse command line arguments
BUILD_METHOD="make"
CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cmake)
            BUILD_METHOD="cmake"
            shift
            ;;
        --make)
            BUILD_METHOD="make"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --make     Use Makefile build (default)"
            echo "  --cmake    Use CMake build"
            echo "  --clean    Clean before building"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Build method: $BUILD_METHOD"

if [ "$BUILD_METHOD" = "make" ]; then
    echo "Building with Makefile..."
    
    if [ "$CLEAN_BUILD" = true ]; then
        echo "Cleaning previous build..."
        make clean
    fi
    
    # Build using Makefile
    make
    
    echo "Running ./rknnops ..."
    ./rknnops
    
elif [ "$BUILD_METHOD" = "cmake" ]; then
    echo "Building with CMake..."
    
    # Create and enter build directory
    BUILD_DIR=build
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    if [ "$CLEAN_BUILD" = true ]; then
        echo "Cleaning previous build..."
        rm -f rknnops CMakeCache.txt
        rm -rf CMakeFiles/
    fi
    
    # Configure and build
    cmake ..
    make
    
    echo "Running ./rknnops ..."
    ./rknnops
    
    # Return to parent directory
    cd ..
fi

echo "=== Build and run completed successfully! ==="