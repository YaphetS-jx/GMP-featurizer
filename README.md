# GMP-Featurizer

A C++ project using the foonathan memory library for efficient memory management.

## Prerequisites

- CMake (version 3.14 or higher)
- C++ compiler with C++17 support
- gemmi CIF parser from https://github.com/project-gemmi/gemmi.git
- nlohmann json parser from https://github.com/nlohmann/json.git

### Or

- use Dockerfile to build the image and launch the container, where all libraries are already installed correctly. 

## Building the Project

To build the project:

1. Make the build script executable:

    `chmod +x ./build.sh`

2. Run the build script with single precision (float): 

    `./build.sh float`

    or with double precision (double):

    `./build.sh double`

3. Run the `gmp-featurizer` with a json configuration file:

    `./build/gmp-featurizer  path/to/json`

4. Run all tests with command in build/ folder, make sure you enable test building in build.sh

    `make test`

### Input JSON file:

1. Required entries

    `system file path` : path to the system file (CIF format)

    `psp file path` : path to the pseudopotential file

    `output file path` : path to the output file

    `feature lists` or both `orders` and `sigmas` : features information

2. Optional entries

    `square` : square of feature option (0 or 1)

    `overlap threshold` : overlap threshold

    `scaling mode` : scaling mode (0 for radial, 1 for both)

    `reference grid` : reference grid (e.g., [10,10,10])

    `num bits per dim` : number of bits per dimension (integer), up to 10

    `num threads` : number of threads (integer)

## Usage Examples

### Basic usage with required parameters:
```json
{
    "system file path": "path/to/system.cif",
    "psp file path": "path/to/pseudopotential.psp",
    "output file path": "path/to/output.txt",
    "orders": [-1, 0, 1, 2],
    "sigmas": [0.1, 0.2, 0.3]
}
```

### Advanced usage with all parameters:
```json
{
    "system file path": "path/to/system.cif",
    "psp file path": "path/to/pseudopotential.psp",
    "output file path": "path/to/output.txt",
    "feature lists": [[1, 0.1], [2, 0.2], [3, 0.3]],
    "square": 1,
    "overlap threshold": 0.5,
    "scaling mode": 1,
    "reference grid": [10, 10, 10],
    "num bits per dim": 8,
    "num threads": 4
}
```

## Command Line Options

You can also use command line options:

    `-h` : Print help message

## Docker Usage

If using Docker:

1. Build the image:
   ```bash
   docker build -t gmp-featurizer .
   ```

2. Run the container:
   ```bash
   docker run -it gmp-featurizer
   ```

3. For development with volume mounting:
   ```bash
   docker run -it -v $(pwd):/app gmp-featurizer
   ```
