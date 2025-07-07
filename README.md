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

### Inside the Container

Once inside either container, you can:

#### Build the Project

```bash
# CPU builds
./build.sh single cpu         # Single precision CPU
./build.sh double cpu         # Double precision CPU

# GPU builds (only in GPU container)
./build.sh single gpu         # Single precision GPU
./build.sh double gpu         # Double precision GPU
```

#### Run Tests

```bash
cd build && make test
```

#### Run the Application

```bash
./build/gmp-featurizer path/to/config.json
```

### Troubleshooting

#### Docker Compose Not Found

If you get "Command 'docker-compose' not found":

```bash
# Modern Docker installations use:
docker compose  # (with space)

# Instead of:
docker-compose  # (with hyphen)
```

#### GPU Not Available

If GPU container fails to start:

```bash
# Check if NVIDIA drivers are working
nvidia-smi

# Verify container has GPU access
docker run --gpus all nvidia/cuda:12.4-base-ubuntu24.04 nvidia-smi
```

#### Permission Issues

If you get permission errors:

```bash
# Add your user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or run:
newgrp docker
```

#### Container Build Failures

```bash
# Clean up and rebuild
docker compose down
docker compose build --no-cache

# For GPU container:
docker compose -f docker-compose.gpu.yml down
docker compose -f docker-compose.gpu.yml build --no-cache
```

### Container Comparison

| Feature | CPU Container | GPU Container |
|---------|---------------|---------------|
| **Base Image** | Ubuntu 24.04 | Ubuntu 24.04 + CUDA 12.4 |
| **Size** | ~500MB | ~2GB |
| **Requirements** | Any Docker | NVIDIA GPU + Drivers |
| **Build Speed** | Fast | Slower (CUDA installation) |
| **Performance** | CPU only | CPU + GPU acceleration |
| **Use Case** | Development, testing | GPU development, production |

### Environment Variables

Both containers set up these environment variables:
- `DEBIAN_FRONTEND=noninteractive`
- Working directory: `/app` (mounted to your project directory)

GPU container additionally sets:
- `CUDA_HOME=/usr/local/cuda`
- `PATH` includes CUDA binaries
- `LD_LIBRARY_PATH` includes CUDA libraries

## GPU Development Environment

The project supports GPU acceleration using CUDA and Thrust for improved performance on compatible systems.

### Overview

The GPU environment provides:
- **CUDA 12.4** with development tools
- **Thrust** library for GPU-accelerated algorithms
- **Full compatibility** with existing CPU code
- **Docker-based** development environment
- **Easy switching** between CPU and GPU builds

### Prerequisites

#### Host System Requirements
- **NVIDIA GPU** with CUDA support
- **NVIDIA drivers** installed on the host
- **Docker** with NVIDIA Container Toolkit

#### Install NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Quick Start

1. **Launch the GPU container** (see Docker Usage section above):
   ```bash
   docker compose -f docker-compose.gpu.yml run --rm gmp-featurizer-gpu bash
   ```

2. **Test the GPU environment**:
   ```bash
   # Build and test with GPU support
   ./build.sh single gpu

   # Run all tests (including CUDA test if GPU is available)
   cd build && make run_all_tests
   ```

This will verify:
- CUDA installation and version
- GPU device availability
- Thrust library availability
- CPU and GPU builds
- Test execution

### Building with GPU Support

#### CPU-Only Builds (in GPU container)

```bash
# Build with single precision (default)
./build.sh                    # Single precision CPU (default)
./build.sh single cpu         # Single precision CPU
./build.sh double cpu         # Double precision CPU
```

#### GPU-Enabled Builds

```bash
# Build with single precision and GPU support
./build.sh single gpu         # Single precision GPU
./build.sh double gpu         # Double precision GPU
```

#### Build Options

The `build.sh` script accepts two parameters:
1. **Floating-point precision**: `single` (default), `double`, `float`
2. **Build type**: `cpu` (default), `gpu`

Examples:
```bash
./build.sh                    # Single precision, CPU only
./build.sh double             # Double precision, CPU only
./build.sh single gpu         # Single precision, GPU enabled
./build.sh double gpu         # Double precision, GPU enabled
```

### Adding GPU Code

#### 1. Create CUDA Source Files

Create `.cu` files for your CUDA kernels:

```cpp
// example_kernel.cu
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void my_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

// Host wrapper function
void gpu_process_data(float* data, int n) {
    thrust::device_vector<float> d_data(data, data + n);
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    my_kernel<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_data.data()), n);
    
    thrust::copy(d_data.begin(), d_data.end(), data);
}
```

#### 2. Conditional Compilation

The CMakeLists.txt already supports CUDA. When `ENABLE_CUDA=ON`:
- CUDA language is enabled
- CUDA include directories are added
- CUDA libraries are linked
- `GMP_ENABLE_CUDA` macro is defined

Use the `GMP_ENABLE_CUDA` macro for conditional compilation:

```cpp
#ifdef GMP_ENABLE_CUDA
    // GPU code here
    gpu_process_data(data, size);
#else
    // CPU fallback
    cpu_process_data(data, size);
#endif
```

### Using Thrust

Thrust is a C++ template library for CUDA that provides:
- **Device vectors**: `thrust::device_vector<T>`
- **Host vectors**: `thrust::host_vector<T>`
- **Algorithms**: `thrust::sort`, `thrust::reduce`, `thrust::transform`
- **Iterators**: `thrust::counting_iterator`, `thrust::transform_iterator`

Example:
```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

// GPU-accelerated vector addition
thrust::device_vector<float> a(1000, 1.0f);
thrust::device_vector<float> b(1000, 2.0f);
thrust::device_vector<float> c(1000);

thrust::transform(a.begin(), a.end(), b.begin(), c.begin(), thrust::plus<float>());
```

### Environment Variables

The container sets up these environment variables:
- `CUDA_HOME=/usr/local/cuda`
- `PATH` includes CUDA binaries
- `LD_LIBRARY_PATH` includes CUDA libraries

### Troubleshooting

#### No CUDA Devices Found
```bash
# Check if NVIDIA drivers are working
nvidia-smi

# Verify container has GPU access
docker run --gpus all nvidia/cuda:12.4-base-ubuntu24.04 nvidia-smi
```

#### Build Failures
```bash
# Check CUDA version compatibility
nvcc --version

# Verify CMake CUDA support
cmake --version
```

#### Performance Issues
- Ensure you're using the correct CUDA architecture for your GPU
- Check GPU memory usage with `nvidia-smi`
- Use `cuda-memcheck` for debugging memory issues

### File Structure

```
GMP-featurizer/
├── Dockerfile.gpu              # GPU-enabled Dockerfile
├── docker-compose.gpu.yml      # Docker Compose configuration
├── build.sh                   # Build script
├── test/                      # Test directory
│   ├── test-cuda.cu           # CUDA test (only built when CUDA enabled)
│   └── ...                    # Other test files
├── CMakeLists.txt             # Updated with CUDA support
└── README.md                  # This file
```

### Next Steps

1. **Test the environment**: Build with `./build.sh single gpu`
2. **Build existing code**: Use `./build.sh single cpu`
3. **Add GPU kernels**: Create `.cu` files and use conditional compilation
4. **Optimize performance**: Profile with `nvprof` or Nsight Systems
5. **Run tests**: Use `cd build && make run_all_tests`

### Support

For issues with the GPU environment:
1. Check the troubleshooting section above
2. Verify your NVIDIA drivers and Docker setup
3. Build with GPU support to identify specific problems
4. Check CUDA and Thrust documentation for programming issues
