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

2. Build the main C++ executable:

    **Basic usage:**
    ```bash
    # Single precision (default)
    ./build.sh

    # Double precision
    ./build.sh --float-type double
    ```

    **GPU support:**
    ```bash
    # GPU with single precision
    ./build.sh --float-type float --build-type gpu

    # GPU with double precision and specific CUDA architecture
    ./build.sh --float-type double --build-type gpu --cuda-arch 120
    ```

    **Parallel compilation:**
    ```bash
    # Use all available threads (default)
    ./build.sh --parallel true

    # Single-threaded compilation
    ./build.sh --parallel false
    ```

3. Build the Python interface:

    **CPU-only Python interface:**
    ```bash
    ./build.sh --python --float-type float
    ```

    **GPU Python interface:**
    ```bash
    ./build.sh --python --float-type float --build-type gpu --cuda-arch 120
    ```

4. Run the `gmp-featurizer` with a json configuration file:

    `./build/gmp-featurizer path/to/json`

5. Run all tests with command in build/ folder, make sure you enable test building in build.sh

    `make test`

### Build Script Options

The `./build.sh` script supports the following options (can be used in any order):

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--float-type TYPE` | `-flt` | Floating point precision: `float` or `double` | `float` |
| `--build-type TYPE` | `-bt` | Build type: `cpu` or `gpu` | `cpu` |
| `--cuda-arch ARCH` | `-ca` | CUDA architecture (e.g., `120`, `130`) | `120` |
| `--ptxas-verbose BOOL` | `-pv` | Enable PTXAS verbose output: `true` or `false` | `false` |
| `--parallel BOOL` | `-p` | Parallel compilation: `true` or `false` | `true` |
| `--python` | `-py` | Build Python interface | `false` |
| `--help` | `-h` | Show help message | - |

**Examples:**
```bash
# Show help
./build.sh --help

# GPU build with custom CUDA arch
./build.sh -bt gpu -ca 130

# Python interface with double precision
./build.sh -py -flt double

# Single-threaded CPU build
./build.sh -p false
```

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

    `enable gpu` : enable GPU acceleration (true or false) - only available in GPU builds

## Python Interface

The Python interface provides automatic resource management and simplified usage:

### Installation

Build the Python interface:
```bash
# CPU-only interface
./build.sh --python --float-type float

# GPU interface
./build.sh --python --float-type float --build-type gpu --cuda-arch 120
```

### Usage

```python
import sys
import os
sys.path.insert(0, '/app/build/python')
import gmp_featurizer

# GPU resources are automatically initialized on import
# and cleaned up on script exit - no manual calls needed!

# Compute features from JSON configuration
features = gmp_featurizer.compute_features('config.json')
print(f"Features shape: {features.shape}")

# The interface returns a NumPy array with shape (n_positions, n_features)
# for GPU mode or (n_positions, n_features) for CPU mode
```

### Python Interface Features

- **Automatic Resource Management**: GPU/CPU resources are initialized on import and cleaned up on exit
- **No Manual Cleanup**: No need to call `initialize_gpu()` or `cleanup()` manually
- **JSON Configuration**: Uses the same JSON configuration format as the C++ executable
- **NumPy Integration**: Returns results as NumPy arrays for easy integration with Python data science tools
- **Error Handling**: Graceful error handling with informative error messages

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
    "num threads": 4,
    "enable gpu": true
}
```

### GPU-specific usage:
```json
{
    "system file path": "path/to/system.cif",
    "psp file path": "path/to/pseudopotential.psp",
    "output file path": "path/to/output.txt",
    "orders": [0, 1, 2],
    "sigmas": [0.1, 0.2, 0.3],
    "enable gpu": true,
    "reference grid": [16, 16, 16]
}
```

## Command Line Options

You can also use command line options:

    `-h` : Print help message

## Docker Usage

### Prerequisites

- **Docker** installed on your system
- For GPU support: **NVIDIA GPU** with **NVIDIA Container Toolkit**

### CPU Container

```bash
# Build and run CPU container
docker compose run --rm gmp-featurizer-cpu bash
```

### GPU Container

```bash
# Build and run GPU container
docker compose -f docker-compose.gpu.yml run --rm gmp-featurizer-gpu bash
```

### Verify GPU Access (inside GPU container)

```bash
# Check GPU availability
nvidia-smi

# Check CUDA version
nvcc --version
```