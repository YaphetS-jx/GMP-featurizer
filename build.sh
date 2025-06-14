# Create a build directory
mkdir -p build
cd build

# Configure the project with CMake
# change the path to the boost and json include directories
# download boost and json from https://github.com/boostorg/pool and https://github.com/nlohmann/json
# then set the path to the boost and json include directories
# then run the following command


# cmake -DBUILD_TESTS=ON \
#   -DBOOST_POOL_INCLUDE_DIR=/path/to/boost \
#   -DNLOHMANN_JSON_INCLUDE_DIR=/path/to/json \
#   -DGEMMI_INCLUDE_DIR=/path/to/gemmi ..

cmake -DCMAKE_CXX_FLAGS="-O0 -pg -g" -DBUILD_TESTS=ON \
    -DBOOST_POOL_INCLUDE_DIR=/media/xx/LEAVE/coding/boost/libs/pool/include \
    -DNLOHMANN_JSON_INCLUDE_DIR=/media/xx/LEAVE/coding/json/include \
    -DGEMMI_INCLUDE_DIR=/media/xx/LEAVE/coding/gemmi/include \
    ..

# Build the project
make
