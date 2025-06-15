# Create a build directory
mkdir -p build
cd build

# Configure the project with CMake
# install boost by apt-get install libboost-all-dev
# change the path to the json and gemmi include directories
# then run the following command

# change here if you want to replace the external libraries
# cmake -DBUILD_TESTS=ON \
#   -DNLOHMANN_JSON_INCLUDE_DIR=/home/xx/Desktop/coding/json/include \
#   -DGEMMI_INCLUDE_DIR=/home/xx/Desktop/coding/gemmi/include ..

cmake -DBUILD_TESTS=ON ..

# Build the project
make
