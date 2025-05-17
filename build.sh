# Create a build directory
mkdir -p build
cd build

# Configure the project with CMake
cmake -DBUILD_TESTS=ON ..

# Build the project
make
