#!/bin/bash

# Unified build script for GMP Featurizer with GPU support
# Usage: 
#   ./build.sh                    # Build main project with single precision (default)
#   ./build.sh double             # Build main project with double precision
#   ./build.sh single             # Build main project with single precision
#   ./build.sh float              # Build main project with single precision (alias)

set -e

# Parse command line arguments
FLOAT_TYPE="${1:-single}"
BUILD_TYPE="${2:-cpu}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to validate floating-point precision
validate_float_type() {
    local float_type="$1"
    case "${float_type,,}" in
        "double"|"d64")
            USE_SINGLE_PRECISION="OFF"
            echo "Building with double precision (double)"
            ;;
        "single"|"float"|"f32")
            USE_SINGLE_PRECISION="ON"
            echo "Building with single precision (float)"
            ;;
        *)
            print_error "Invalid floating-point type '$float_type'"
            echo "Valid options: double, single, float"
            echo "Usage: ./build.sh [double|single|float] [cpu|gpu]"
            exit 1
            ;;
    esac
}

# Function to validate build type
validate_build_type() {
    local build_type="$1"
    case "${build_type,,}" in
        "cpu")
            ENABLE_CUDA="OFF"
            echo "Building CPU-only version"
            ;;
        "gpu")
            ENABLE_CUDA="ON"
            echo "Building with GPU support"
            ;;
        *)
            print_error "Invalid build type '$build_type'"
            echo "Valid options: cpu, gpu"
            echo "Usage: ./build.sh [double|single|float] [cpu|gpu]"
            exit 1
            ;;
    esac
}

# Function to check CUDA availability
check_cuda() {
    if [ "$ENABLE_CUDA" = "ON" ]; then
        if ! command -v nvcc &> /dev/null; then
            print_error "nvcc not found. CUDA is required for GPU builds."
            echo "Make sure you're running this in a CUDA-enabled container or have CUDA installed."
            exit 1
        fi
        
        # Get CUDA version
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_status "CUDA version: $CUDA_VERSION"
    fi
}

# Function to build main project
build_main_project() {
    print_status "Building GMP Featurizer project..."
    
    validate_float_type "$FLOAT_TYPE"
    validate_build_type "$BUILD_TYPE"
    check_cuda
    
    BUILD_DIR="build"
    
    # Create a build directory
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    
    # Configure the project with CMake
    print_status "Configuring CMake with:"
    echo "  - Floating point precision: $FLOAT_TYPE"
    echo "  - Build type: $BUILD_TYPE"
    echo "  - CUDA enabled: $ENABLE_CUDA"
    
    cmake -DBUILD_TESTS=ON -DBUILD_TYPE_RELEASE=ON \
      -DUSE_SINGLE_PRECISION=$USE_SINGLE_PRECISION \
      -DENABLE_CUDA=$ENABLE_CUDA \
      ..
    
    # Build the project
    print_status "Building project..."
    make -j
    
    print_status "Build completed successfully!"
    echo "Floating-point type: $FLOAT_TYPE"
    echo "Build type: $BUILD_TYPE"
    echo "CUDA enabled: $ENABLE_CUDA"
    
    cd ..
}

# Main execution logic
build_main_project
