# Create a build directory
mkdir -p build
cd build

# Configure the project with CMake
# install boost by apt-get install libboost-all-dev
# change the path to the json and gemmi include directories
# then run the following command

# change here if you want to replace the external libraries
# For profiling: use -DBUILD_TYPE_PROFILE=ON instead of -DBUILD_TYPE_RELEASE=ON
# cmake -DBUILD_TESTS=ON -DBUILD_TYPE_RELEASE=ON \
#   -DGEMMI_INCLUDE_DIR=/home/xx/Desktop/coding/gemmi/include\
#   -DXSIMD_INCLUDE_DIR=/home/xx/Desktop/coding/xsimd/include ..

cmake ..

# Build the project
make -j
