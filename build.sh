#!/bin/bash

# Unified build script for GMP Featurizer with GPU and Python support
# Usage: 
#   ./build.sh                                    # Build main project with defaults
#   ./build.sh -flt double                        # Build with double precision
#   ./build.sh -bt gpu                            # Build with GPU support
#   ./build.sh -py                                # Build Python interface
#   ./build.sh -py -bt gpu                        # Build Python interface with GPU
#   ./build.sh -ca 120 -pv                        # Build with specific CUDA arch and verbose PTXAS
#   ./build.sh -p false                           # Single-threaded compilation
#   ./build.sh -h                                 # Show help

set -e

# Default values
FLOAT_TYPE="single"
BUILD_TYPE="cpu"
CUDA_ARCH="120"
PTXAS_VERBOSE="false"
PARALLEL="true"
BUILD_PYTHON="false"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --float-type|-flt)
            FLOAT_TYPE="$2"
            shift 2
            ;;
        --build-type|-bt)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --cuda-arch|-ca)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --ptxas-verbose|-pv)
            PTXAS_VERBOSE="true"
            shift
            ;;
        --parallel|-p)
            PARALLEL="$2"
            shift 2
            ;;
        --python|-py)
            BUILD_PYTHON="true"
            shift
            ;;
        --help|-h)
            echo "GMP Featurizer Build Script"
            echo "============================"
            echo ""
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --float-type, -flt TYPE      Floating point precision (single|double) [default: single]"
            echo "  --build-type, -bt TYPE       Build type (cpu|gpu) [default: cpu]"
            echo "  --cuda-arch, -ca ARCH        CUDA architecture (50,60,70,75,80,86,89,120,etc) [default: 120]"
            echo "  --ptxas-verbose, -pv         Enable PTXAS verbose output"
            echo "  --parallel, -p BOOL          Enable parallel compilation (true|false) [default: true]"
            echo "  --python, -py                Build Python interface"
            echo "  --help, -h                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./build.sh                                    # Default build"
            echo "  ./build.sh -flt double -bt gpu                # GPU build with double precision"
            echo "  ./build.sh --python --build-type gpu          # Python interface with GPU"
            echo "  ./build.sh -ca 80 -pv                         # Specific CUDA arch with verbose output"
            echo "  ./build.sh -p false                           # Single-threaded compilation"
            echo "  ./build.sh -py -bt gpu                        # Python interface with GPU (short form)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done 

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

# Function to validate parallel option
validate_parallel() {
    local parallel_opt="$1"
    case "${parallel_opt,,}" in
        "true"|"on"|"yes"|"1")
            MAKE_THREADS="-j"
            echo "Parallel compilation enabled (all threads)"
            ;;
        "false"|"off"|"no"|"0")
            MAKE_THREADS=""
            echo "Single-threaded compilation enabled"
            ;;
        *)
            print_error "Invalid parallel option '$parallel_opt'"
            echo "Valid options: true, false, on, off, yes, no, 1, 0"
            echo "Usage: ./build.sh --parallel [true|false]"
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
    validate_parallel "$PARALLEL"
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
    echo "  - Parallel compilation: $PARALLEL"
    
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
    echo "Parallel compilation: $PARALLEL"
    
    cd ..
}

# Function to build Python interface
build_python_interface() {
    print_status "Building GMP Featurizer Python interface..."
    
    validate_float_type "$FLOAT_TYPE"
    validate_build_type "$BUILD_TYPE"
    validate_cuda_arch "$CUDA_ARCH"
    validate_ptxas_verbose "$PTXAS_VERBOSE"
    validate_parallel "$PARALLEL"
    check_cuda
    
    BUILD_DIR="build"
    
    # Create a build directory
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    
    # Configure the project with CMake for Python module
    print_status "Configuring CMake for Python module with:"
    echo "  - Floating point precision: $FLOAT_TYPE"
    echo "  - Build type: $BUILD_TYPE"
    echo "  - CUDA enabled: $ENABLE_CUDA"
    echo "  - CUDA architecture: $CUDA_ARCH"
    echo "  - PTXAS verbose: $ENABLE_PTXAS_VERBOSE"
    echo "  - Parallel compilation: $PARALLEL"
    
    cmake -DBUILD_PYTHON_MODULE=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DUSE_SINGLE_PRECISION=$USE_SINGLE_PRECISION \
      -DENABLE_CUDA=$ENABLE_CUDA \
      -DCUDA_ARCH=$CUDA_ARCH \
      -DENABLE_PTXAS_VERBOSE=$ENABLE_PTXAS_VERBOSE \
      -DGEMMI_INCLUDE_DIR=/usr/local/src/gemmi/include \
      -DNLOHMANN_JSON_INCLUDE_DIR=/usr/local/src/json/single_include \
      ..
    
    # Build the Python module
    print_status "Building Python module..."
    make gmp_featurizer $MAKE_THREADS
    
    # Check if the module was created successfully
    if ls python/gmp_featurizer*.so 1> /dev/null 2>&1; then
        MODULE_FILE=$(ls python/gmp_featurizer*.so | head -1)
        print_status "Python module built successfully!"
        echo "  Location: $(pwd)/$MODULE_FILE"
        echo ""
        echo "To use the module:"
        echo "1. Add the build directory to your Python path:"
        echo "   export PYTHONPATH=\"$(pwd)/python:\$PYTHONPATH\""
        echo ""
        echo "2. Test the interface:"
        echo "   cd ../example"
        echo "   python3 -c \"import sys; sys.path.insert(0, '/app/build/python'); import gmp_featurizer; print('Import successful')\""
    else
        print_error "Failed to build Python module"
        exit 1
    fi
    
    cd ..
}

# Main execution logic
if [ "$BUILD_PYTHON" = "true" ]; then
    build_python_interface
else
    build_main_project
fi
