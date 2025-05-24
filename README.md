# GMP-Featurizer

A C++ project using the foonathan memory library for efficient memory management.

## Prerequisites

- CMake (version 3.14 or higher)
- C++ compiler with C++11 support
- boost/pool memory management from https://github.com/boostorg/pool
- nlohmann json parser from https://github.com/nlohmann/json

## Building the Project

To build the project:

1. Make the build script executable:

    `chmod +x ./build.sh`

2. Run the build script: 

    `./build.sh`

3. Run the `gmp-featurizer` with a json configuration file:

    `./build/gmp-featurizer  path/to/json`

4. Run all tests with command in build/ folder 

    `make test`



