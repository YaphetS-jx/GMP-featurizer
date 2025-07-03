#!/bin/bash

# Build script for GMP Featurizer with floating-point precision selection
# Usage: 
#   ./build.sh                    # Build with single precision (default)
#   ./build.sh double             # Build with double precision
#   ./build.sh single             # Build with single precision
#   ./build.sh float              # Build with single precision (alias)

set -e

# Parse command line arguments
FLOAT_TYPE="${1:-single}"
BUILD_DIR="build"

# Validate input
case "${FLOAT_TYPE,,}" in
    "double"|"d64")
        USE_SINGLE_PRECISION="OFF"
        echo "Building with double precision (double)"
        ;;
    "single"|"float"|"f32")
        USE_SINGLE_PRECISION="ON"
        echo "Building with single precision (float)"
        ;;
    *)
        echo "Error: Invalid floating-point type '$FLOAT_TYPE'"
        echo "Valid options: double, single, float"
        echo "Usage: ./build.sh [double|single|float]"
        exit 1
        ;;
esac

# Create a build directory
mkdir -p build
cd build

# Configure the project with CMake
# change the path to the json and gemmi include directories
# then run the following command

# change here if you want to replace the external libraries
# For profiling: use -DBUILD_TYPE_PROFILE=ON instead of -DBUILD_TYPE_RELEASE=ON
# cmake -DBUILD_TESTS=ON -DBUILD_TYPE_RELEASE=ON \
#   -DUSE_SINGLE_PRECISION=$USE_SINGLE_PRECISION \
#   -DGEMMI_INCLUDE_DIR=/home/xx/Desktop/coding/gemmi/include \
#   -DNLOHMANN_JSON_INCLUDE_DIR=/home/xx/Desktop/coding/json/single_include \
#   ..

cmake -DBUILD_TESTS=ON -DBUILD_TYPE_RELEASE=ON \
  -DUSE_SINGLE_PRECISION=$USE_SINGLE_PRECISION \
  ..

# Build the project
make -j

echo "Build completed successfully!"
echo "Floating-point type: $FLOAT_TYPE"
