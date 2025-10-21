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
    py::array_t<gmp::gmp_float> vector2d_to_numpy(const std::vector<std::vector<gmp::gmp_float>>& data) {
        if (data.empty()) {
            return py::array_t<gmp::gmp_float>(std::vector<size_t>{0, 0});
        }
        
        size_t rows = data.size();
        size_t cols = data[0].size();
        
        // Create numpy array
        auto result = py::array_t<gmp::gmp_float>(std::vector<size_t>{rows, cols});
        auto buf = result.mutable_unchecked<2>();
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                buf(i, j) = data[i][j];
            }
        }
        
        return result;
    }
    
    // Helper function to convert 1D vector to numpy array (for GPU output)
    template<typename Allocator>
    py::array_t<gmp::gmp_float> vector1d_to_numpy(const std::vector<gmp::gmp_float, Allocator>& data, 
                                        size_t n_positions, size_t n_features) {
        if (data.empty()) {
            return py::array_t<gmp::gmp_float>(std::vector<size_t>{0, 0});
        }
        
        // Reshape 1D data to 2D: [n_features, n_positions]
        auto result = py::array_t<gmp::gmp_float>(std::vector<size_t>{n_positions, n_features});
        auto buf = result.mutable_unchecked<2>();
        
        for (size_t j = 0; j < n_features; ++j) {
            for (size_t i = 0; i < n_positions; ++i) {
                buf(i, j) = data[j * n_positions + i];
            }
        }
        
        return result;
    }

    // Function that takes JSON file path and returns numpy array
    py::array_t<gmp::gmp_float> compute_features_from_json(const std::string& json_file) {
        try {
            // Reset error state
            gmp::update_error(gmp::error_t::success);
            
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
                
                // Get dimensions for numpy conversion using the same logic as the featurizer
                gmp::containers::vector<gmp::geometry::point3d_t<gmp_float>> ref_positions;
                if (!input->files->get_reference_grid_file().empty()) {
                    auto temp_ref_positions = input->read_reference_grid_from_file(input->files->get_reference_grid_file());
                    ref_positions.assign(temp_ref_positions.begin(), temp_ref_positions.end());
                } else {
                    // Create unit_cell to get atoms for set_ref_positions
                    std::unique_ptr<gmp::atom::unit_cell_flt> unit_cell = std::make_unique<gmp::atom::unit_cell_flt>(input->files->get_atom_file());
                    auto ref_grid = input->descriptor_config->get_ref_grid();
                    ref_positions = gmp::atom::set_ref_positions(ref_grid, unit_cell->get_atoms());
                }
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

    // Function that accepts parameters directly instead of JSON file
    py::array_t<gmp::gmp_float> compute_features(
        const std::string& atom_file,
        const std::string& psp_file,
        const std::string& output_file = "./gmpFeatures.dat",
        const std::vector<int>& orders = {},
        const std::vector<double>& sigmas = {},
        const std::vector<std::pair<int, double>>& feature_lists = {},
        bool square = false,
        double overlap_threshold = 1e-11,
        int scaling_mode = 0,
        const std::vector<int>& uniform_reference_grid = {16, 16, 16},
        py::array_t<double> reference_grid = py::array_t<double>(),
        int num_bits_per_dim = 5,
        int num_threads = 0,
        bool enable_gpu = true
    ) {
        try {
            // Reset error state
            gmp::update_error(gmp::error_t::success);
            
            // Create input object with default constructor
            std::unique_ptr<input::input_t> input = std::make_unique<input::input_t>();
            GMP_CHECK(get_last_error());
            
            // Set file paths
            input->files->set_atom_file(atom_file);
            input->files->set_psp_file(psp_file);
            input->files->set_output_file(output_file);
            
            // Set descriptor configuration
            input->descriptor_config->set_square(square);
            input->descriptor_config->set_overlap_threshold(static_cast<gmp::gmp_float>(overlap_threshold));
            input->descriptor_config->set_scaling_mode(static_cast<input::scaling_mode_t>(scaling_mode));
            input->descriptor_config->set_num_bits_per_dim(static_cast<uint8_t>(num_bits_per_dim));
            input->descriptor_config->set_num_threads(static_cast<size_t>(num_threads));
            input->descriptor_config->set_enable_gpu(enable_gpu);
            
            // Set reference grid
            if (uniform_reference_grid.size() == 3) {
                input::array3d_int32 ref_grid_array{
                    static_cast<int32_t>(uniform_reference_grid[0]),
                    static_cast<int32_t>(uniform_reference_grid[1]),
                    static_cast<int32_t>(uniform_reference_grid[2])
                };
                input->descriptor_config->set_ref_grid(ref_grid_array);
            }
            
            // Set reference positions in input object if provided
            if (reference_grid.size() > 0) {
                // Use provided reference grid points directly
                auto buf = reference_grid.unchecked<2>();
                size_t n_points = buf.shape(0);
                gmp::containers::vector<gmp::geometry::point3d_t<gmp_float>> ref_positions;
                ref_positions.reserve(n_points);
                for (size_t i = 0; i < n_points; ++i) {
                    ref_positions.push_back(gmp::geometry::point3d_t<gmp_float>{
                        static_cast<gmp::gmp_float>(buf(i, 0)),
                        static_cast<gmp::gmp_float>(buf(i, 1)),
                        static_cast<gmp::gmp_float>(buf(i, 2))
                    });
                }
                input->set_reference_positions(ref_positions);
            }
            
            // Set feature list
            std::vector<int> orders_vec = orders;
            std::vector<gmp::gmp_float> sigmas_vec;
            std::vector<std::tuple<int, gmp::gmp_float>> feature_list_vec;
            
            // Convert sigmas to gmp_float
            for (double sigma : sigmas) {
                sigmas_vec.push_back(static_cast<gmp::gmp_float>(sigma));
            }
            
            // Convert feature lists to gmp_float
            for (const auto& pair : feature_lists) {
                feature_list_vec.emplace_back(pair.first, static_cast<gmp::gmp_float>(pair.second));
            }
            
            // If no feature lists provided, use default values
            if (orders_vec.empty() && sigmas_vec.empty() && feature_list_vec.empty()) {
                // Use default values from the original config.json
                orders_vec = {0, 1, 2, 3};
                sigmas_vec = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
            }
            
            input->descriptor_config->set_feature_list(orders_vec, sigmas_vec, feature_list_vec);
            
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
                
                // Get dimensions for numpy conversion using the same logic as the featurizer
                gmp::containers::vector<gmp::geometry::point3d_t<gmp_float>> ref_positions;
                if (input->has_reference_positions()) {
                    // Use provided reference positions directly (no copying needed)
                    ref_positions = input->get_reference_positions();
                } else {
                    // Create unit_cell to get atoms for set_ref_positions
                    std::unique_ptr<gmp::atom::unit_cell_flt> unit_cell = std::make_unique<gmp::atom::unit_cell_flt>(input->files->get_atom_file());
                    auto ref_grid = input->descriptor_config->get_ref_grid();
                    ref_positions = gmp::atom::set_ref_positions(ref_grid, unit_cell->get_atoms());
                }
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

// add the function to change reference grid
// add the option not to take weighted sum 

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
    // Primary interface - direct parameter specification
    m.def("compute_features", &gmp::python::compute_features,
          "Compute features with direct parameter specification (auto-initializes resources)",
          py::arg("atom_file"),
          py::arg("psp_file"),
          py::arg("output_file") = "./gmpFeatures.dat",
          py::arg("orders") = std::vector<int>{},
          py::arg("sigmas") = std::vector<double>{},
          py::arg("feature_lists") = std::vector<std::pair<int, double>>{},
          py::arg("square") = false,
          py::arg("overlap_threshold") = 1e-11,
          py::arg("scaling_mode") = 0,
          py::arg("uniform_reference_grid") = std::vector<int>{16, 16, 16},
          py::arg("reference_grid") = py::array_t<double>(),
          py::arg("num_bits_per_dim") = 5,
          py::arg("num_threads") = 0,
          py::arg("enable_gpu") = true);
    
    // Legacy interface - JSON-based (for backward compatibility)
    m.def("compute_features_from_json", &gmp::python::compute_features_from_json,
          "Compute features from JSON configuration file (auto-initializes resources) - Legacy interface",
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
