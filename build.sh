#!/bin/bash

# Unified build script for GMP Featurizer with GPU support
# Usage: 
#   ./build.sh                    # Build main project with single precision (default)
#   ./build.sh double             # Build main project with double precision
#   ./build.sh single             # Build main project with single precision
#   ./build.sh float              # Build main project with single precision (alias)
#   ./build.sh single gpu         # Build with GPU support
#   ./build.sh single gpu 120     # Build with GPU support and specific architecture
#   ./build.sh single gpu 120 ptxas    # Build with GPU support and PTXAS verbose output
#   ./build.sh single gpu 120 ptxas single    # Build with single-threaded compilation

set -e

# Parse command line arguments
FLOAT_TYPE="${1:-single}"
BUILD_TYPE="${2:-cpu}"
CUDA_ARCH="${3:-120}"  # Default to sm_120 
PTXAS_VERBOSE="${4:-false}"  # Default to false
THREADS="${5:-auto}"  # Default to auto (parallel) 

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
            echo "Usage: ./build.sh [double|single|float] [cpu|gpu] [architecture] [ptxas] [threads]"
            exit 1
            ;;
    esac
}

# Function to validate CUDA architecture
validate_cuda_arch() {
    local arch="$1"
    if [ "$ENABLE_CUDA" = "ON" ]; then
        # Common CUDA architectures
        case "$arch" in
            "50"|"52"|"53"|"60"|"61"|"62"|"70"|"72"|"75"|"80"|"86"|"87"|"89"|"90"|"100"|"101"|"120")
                echo "Using CUDA architecture: sm_$arch"
                ;;
            *)
                print_warning "Unknown CUDA architecture '$arch'. Proceeding anyway..."
                echo "Common architectures: 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 120"
                ;;
        esac
    fi
}

# Function to validate PTXAS verbose option
validate_ptxas_verbose() {
    local ptxas_opt="$1"
    case "${ptxas_opt,,}" in
        "true"|"on"|"yes"|"ptxas"|"verbose")
            ENABLE_PTXAS_VERBOSE="ON"
            if [ "$ENABLE_CUDA" = "ON" ]; then
                echo "PTXAS verbose output enabled"
            else
                print_warning "PTXAS verbose option ignored (CUDA not enabled)"
                ENABLE_PTXAS_VERBOSE="OFF"
            fi
            ;;
        "false"|"off"|"no"|"")
            ENABLE_PTXAS_VERBOSE="OFF"
            ;;
        *)
            print_error "Invalid PTXAS verbose option '$ptxas_opt'"
            echo "Valid options: true, false, on, off, yes, no, ptxas, verbose"
            echo "Usage: ./build.sh [double|single|float] [cpu|gpu] [architecture] [ptxas] [threads]"
            exit 1
            ;;
    esac
}

# Function to validate threads option
validate_threads() {
    local threads_opt="$1"
    case "${threads_opt,,}" in
        "single"|"1"|"serial")
            MAKE_THREADS="-j1"
            echo "Single-threaded compilation enabled"
            ;;
        "auto"|"parallel"|"")
            MAKE_THREADS="-j"
            echo "Parallel compilation enabled (auto)"
            ;;
        [0-9]*)
            MAKE_THREADS="-j$threads_opt"
            echo "Compilation with $threads_opt threads"
            ;;
        *)
            print_error "Invalid threads option '$threads_opt'"
            echo "Valid options: single, 1, auto, parallel, or a specific number (e.g., 4, 8)"
            echo "Usage: ./build.sh [double|single|float] [cpu|gpu] [architecture] [ptxas] [threads]"
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
    validate_cuda_arch "$CUDA_ARCH"
    validate_ptxas_verbose "$PTXAS_VERBOSE"
    validate_threads "$THREADS"
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
    echo "  - CUDA architecture: $CUDA_ARCH"
    echo "  - PTXAS verbose: $ENABLE_PTXAS_VERBOSE"
    echo "  - Compilation threads: $MAKE_THREADS"
    
    cmake -DBUILD_TESTS=ON -DBUILD_TYPE_RELEASE=ON \
      -DUSE_SINGLE_PRECISION=$USE_SINGLE_PRECISION \
      -DENABLE_CUDA=$ENABLE_CUDA \
      -DCUDA_ARCH=$CUDA_ARCH \
      -DENABLE_PTXAS_VERBOSE=$ENABLE_PTXAS_VERBOSE \
      ..
    
    # Build the project
    print_status "Building project..."
    make $MAKE_THREADS
    
    print_status "Build completed successfully!"
    echo "Floating-point type: $FLOAT_TYPE"
    echo "Build type: $BUILD_TYPE"
    echo "CUDA enabled: $ENABLE_CUDA"
    echo "CUDA architecture: $CUDA_ARCH"
    echo "PTXAS verbose: $ENABLE_PTXAS_VERBOSE"
    echo "Compilation threads: $MAKE_THREADS"
    
    cd ..
}

# Main execution logic
build_main_project
