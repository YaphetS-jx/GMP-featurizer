#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <signal.h>
#include <atomic>

// Include the main featurizer headers
#include "input.hpp"
#include "cpu_featurizer.hpp"
#include "gpu_featurizer.hpp"
#include "atom.hpp"
#include "featurizer.hpp"
#include "error.hpp"
#ifdef GMP_ENABLE_CUDA
#include "resources.hpp"
#include <cuda_runtime.h>
#endif

namespace py = pybind11;

// Global state management for automatic initialization/cleanup
static std::atomic<bool> gpu_initialized{false};
static std::atomic<bool> cpu_initialized{false};
static std::atomic<bool> cleanup_registered{false};

// Forward declarations
namespace gmp { namespace python {
    void initialize_gpu();
    void initialize_cpu();
    void cleanup();
}}

// Global functions for automatic initialization/cleanup
void ensure_gpu_initialized() {
    if (!gpu_initialized.load()) {
        try {
            gmp::python::initialize_gpu();
            gpu_initialized = true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: GPU initialization failed: " << e.what() << std::endl;
        }
    }
}

void ensure_cpu_initialized() {
    if (!cpu_initialized.load()) {
        try {
            gmp::python::initialize_cpu();
            cpu_initialized = true;
        } catch (const std::exception& e) {
            std::cerr << "Warning: CPU initialization failed: " << e.what() << std::endl;
        }
    }
}

void auto_cleanup_handler() {
    try {
        if (gpu_initialized.load()) {
            gmp::python::cleanup();
            gpu_initialized = false;
        }
    } catch (...) {
        // Ignore cleanup errors during program exit
    }
}

void signal_handler(int signal) {
    auto_cleanup_handler();
    std::exit(signal);
}

namespace gmp { namespace python {

    // Helper function to convert 2D vector to numpy array
    py::array_t<double> vector2d_to_numpy(const std::vector<std::vector<gmp::gmp_float>>& data) {
        if (data.empty()) {
            return py::array_t<double>(std::vector<size_t>{0, 0});
        }
        
        size_t rows = data.size();
        size_t cols = data[0].size();
        
        // Create numpy array
        auto result = py::array_t<double>(std::vector<size_t>{rows, cols});
        auto buf = result.mutable_unchecked<2>();
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                buf(i, j) = static_cast<double>(data[i][j]);
            }
        }
        
        return result;
    }
    
    // Helper function to convert 1D vector to numpy array (for GPU output)
    template<typename Allocator>
    py::array_t<double> vector1d_to_numpy(const std::vector<gmp::gmp_float, Allocator>& data, 
                                        size_t n_positions, size_t n_features) {
        if (data.empty()) {
            return py::array_t<double>(std::vector<size_t>{0, 0});
        }
        
        // Reshape 1D data to 2D: [n_features, n_positions]
        auto result = py::array_t<double>(std::vector<size_t>{n_positions, n_features});
        auto buf = result.mutable_unchecked<2>();
        
        for (size_t j = 0; j < n_features; ++j) {
            for (size_t i = 0; i < n_positions; ++i) {
                buf(i, j) = static_cast<double>(data[j * n_positions + i]);
            }
        }
        
        return result;
    }

    // Main function that takes JSON file path and returns numpy array
    py::array_t<double> compute_features(const std::string& json_file) {
        try {
            // Parse input
            std::unique_ptr<input::input_t> input = std::make_unique<input::input_t>(json_file);
            GMP_CHECK(get_last_error());
            
            // Check if GPU is enabled and preferred
            bool use_gpu = false;
#ifdef GMP_ENABLE_CUDA
            use_gpu = input->get_descriptor_config()->get_enable_gpu();
#endif

            if (use_gpu) {
#ifdef GMP_ENABLE_CUDA
                ensure_gpu_initialized();
                // GPU featurizer - now returns data
                auto result = gmp::run_gpu_featurizer(input.get());
                GMP_CHECK(get_last_error());
                
                // Get dimensions for numpy conversion
                auto ref_positions = atom::set_ref_positions(input->descriptor_config->get_ref_grid(), 
                                                           atom::unit_cell_flt(input->files->get_atom_file()).get_atoms());
                size_t n_features = input->descriptor_config->get_feature_list().size();
                size_t n_positions = ref_positions.size();
                return vector1d_to_numpy(result, n_positions, n_features);
#else
                throw std::runtime_error("GPU support not compiled in this build");
#endif
            } else {
                ensure_cpu_initialized();
                // CPU featurizer - now returns data
                auto result = gmp::run_cpu_featurizer(input.get());
                GMP_CHECK(get_last_error());
                
                // Convert to numpy array
                return vector2d_to_numpy(result);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Featurizer error: " + std::string(e.what()));
        }
    }


    // Cleanup function for GPU resources (manual control)
    void cleanup() {
#ifdef GMP_ENABLE_CUDA
        try {
            // Synchronize all CUDA operations before cleanup
            cudaDeviceSynchronize();
            gmp::resources::gmp_resource::instance().cleanup();
        } catch (const std::exception& e) {
            // Ignore cleanup errors to prevent core dumps
            std::cerr << "Warning: Error during GPU cleanup: " << e.what() << std::endl;
        }
#endif
    }

    // Initialize GPU resources (manual control)
    void initialize_gpu() {
#ifdef GMP_ENABLE_CUDA
        auto& resource_manager = gmp::resources::gmp_resource::instance();
        resource_manager.get_gpu_device_memory_pool();
        resource_manager.get_pinned_host_memory_pool();
        resource_manager.get_stream();
#endif
    }

    // Initialize CPU resources (manual control)
    void initialize_cpu() {
        // CPU initialization is handled automatically by the thread pool
        // when first accessed, so no explicit initialization needed
    }

}} // namespace gmp::python

// Python module definition
PYBIND11_MODULE(gmp_featurizer, m) {
    m.doc() = "GMP Featurizer Python Interface with Automatic Resource Management";
    
    // Automatically initialize GPU resources on module import
    try {
        ensure_gpu_initialized();
    } catch (const std::exception& e) {
        std::cerr << "Warning: GPU pre-initialization failed: " << e.what() << std::endl;
    }
    
    // Register cleanup handlers for automatic resource management
    std::atexit([]() {
        auto_cleanup_handler();
    });
    
    // Register signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Main functions
    m.def("compute_features", &gmp::python::compute_features,
          "Compute features from JSON configuration file (auto-initializes resources)",
          py::arg("json_file"));
    
    // Manual control functions (optional)
    m.def("cleanup", &gmp::python::cleanup,
          "Manually cleanup GPU resources (automatic cleanup is enabled by default)");
    
    m.def("initialize_gpu", &gmp::python::initialize_gpu,
          "Manually initialize GPU resources (automatic initialization is enabled by default)");
    
    m.def("initialize_cpu", &gmp::python::initialize_cpu,
          "Manually initialize CPU resources (automatic initialization is enabled by default)");
    
    // Add some constants
    m.attr("__version__") = "1.0.0";
    m.attr("__auto_cleanup__") = true;
    m.attr("__auto_init__") = true;
}
