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
#include "math.hpp"
#include "mcsh.hpp"
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

    // FeaturizerContext class to hold initialized data
    // This allows separation of initialization and computation, enabling different initialization strategies
    class FeaturizerContext {
    public:
        // Store input_t to maintain descriptor_config and other settings
        std::unique_ptr<input::input_t> input;
        
        // Store initialization result
        input::featurizer_init_result_t init_result;
        
        // Constructor from input_t (uses file-based initialization)
        FeaturizerContext(std::unique_ptr<input::input_t> input_ptr) 
            : input(std::move(input_ptr)) {
            // Initialize featurizer data using convenience function
            init_result = input::initialize_featurizer(input.get());
            GMP_CHECK(get_last_error());
        }
        
        // Constructor with custom initialization result (for direct data initialization)
        FeaturizerContext(
            std::unique_ptr<input::input_t> input_ptr,
            input::featurizer_init_result_t custom_init_result
        ) : input(std::move(input_ptr)), init_result(std::move(custom_init_result)) {
            // Use provided initialization result
        }
        
        // Get descriptor config
        const input::descriptor_config_flt* get_descriptor_config() const {
            return input->get_descriptor_config();
        }
        
        // Get unit cell
        const atom::unit_cell_flt* get_unit_cell() const {
            return init_result.unit_cell.get();
        }
        
        // Get psp config
        const atom::psp_config_flt* get_psp_config() const {
            return init_result.psp_config.get();
        }
        
        // Get reference positions
        const gmp::containers::vector<gmp::geometry::point3d_t<gmp_float>>& get_ref_positions() const {
            return init_result.ref_positions;
        }
        
        // Check if GPU is enabled
        bool is_gpu_enabled() const {
#ifdef GMP_ENABLE_CUDA
            return input->get_descriptor_config()->get_enable_gpu();
#else
            return false;
#endif
        }
        
        // Get output mode
        bool get_output_raw_data() const {
            return input->get_descriptor_config()->get_output_raw_data();
        }
    };

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
    // Format: [feature][position] -> [position][feature]
    template<typename Allocator>
    py::array_t<gmp::gmp_float> vector1d_to_numpy(const std::vector<gmp::gmp_float, Allocator>& data, 
                                        size_t n_positions, size_t n_features) {
        if (data.empty()) {
            return py::array_t<gmp::gmp_float>(std::vector<size_t>{0, 0});
        }
        
        // Reshape 1D data to 2D: [n_features, n_positions] -> [n_positions, n_features]
        auto result = py::array_t<gmp::gmp_float>(std::vector<size_t>{n_positions, n_features});
        auto buf = result.mutable_unchecked<2>();
        
        for (size_t j = 0; j < n_features; ++j) {
            for (size_t i = 0; i < n_positions; ++i) {
                buf(i, j) = data[j * n_positions + i];
            }
        }
        
        return result;
    }
    
    // Helper function to convert 1D vector to numpy array (for GPU raw data output)
    // Format: [position][features] -> [position][features] (no transpose needed)
    template<typename Allocator>
    py::array_t<gmp::gmp_float> vector1d_to_numpy_raw(const std::vector<gmp::gmp_float, Allocator>& data, 
                                        size_t n_positions, size_t n_raw_features) {
        if (data.empty()) {
            return py::array_t<gmp::gmp_float>(std::vector<size_t>{0, 0});
        }
        
        // Reshape 1D data to 2D: [position][features] format (no transpose)
        auto result = py::array_t<gmp::gmp_float>(std::vector<size_t>{n_positions, n_raw_features});
        auto buf = result.mutable_unchecked<2>();
        
        for (size_t i = 0; i < n_positions; ++i) {
            for (size_t j = 0; j < n_raw_features; ++j) {
                buf(i, j) = data[i * n_raw_features + j];
            }
        }
        
        return result;
    }

    // ============================================================================
    // Helper Functions for Direct Data Initialization
    // ============================================================================
    
    // Helper function to create unit_cell from Python data
    std::unique_ptr<atom::unit_cell_flt> create_unit_cell_from_data(
        py::array_t<double> cell_lengths,      // [a, b, c] in Angstroms
        py::array_t<double> cell_angles,        // [alpha, beta, gamma] in degrees
        py::array_t<double> atom_positions,    // [n_atoms, 3] fractional coordinates
        const std::vector<std::string>& atom_types,   // [n_atoms] element symbols (Python list)
        py::array_t<double> atom_occupancies = py::array_t<double>(), // [n_atoms] optional
        const std::array<bool, 3>& periodicity = {true, true, true}
    ) {
        // Validate input shapes
        if (cell_lengths.size() != 3) {
            throw std::runtime_error("cell_lengths must have exactly 3 elements [a, b, c]");
        }
        if (cell_angles.size() != 3) {
            throw std::runtime_error("cell_angles must have exactly 3 elements [alpha, beta, gamma]");
        }
        
        auto pos_buf = atom_positions.unchecked<2>();
        size_t n_atoms = pos_buf.shape(0);
        
        if (pos_buf.shape(1) != 3) {
            throw std::runtime_error("atom_positions must be [n_atoms, 3] array");
        }
        if (atom_types.size() != n_atoms) {
            throw std::runtime_error("atom_types must have same length as atom_positions");
        }
        
        // Convert cell parameters
        gmp::math::array3d_t<gmp::gmp_float> cell_len = {
            static_cast<gmp::gmp_float>(cell_lengths.at(0)),
            static_cast<gmp::gmp_float>(cell_lengths.at(1)),
            static_cast<gmp::gmp_float>(cell_lengths.at(2))
        };
        gmp::math::array3d_t<gmp::gmp_float> cell_ang = {
            static_cast<gmp::gmp_float>(cell_angles.at(0)),
            static_cast<gmp::gmp_float>(cell_angles.at(1)),
            static_cast<gmp::gmp_float>(cell_angles.at(2))
        };
        
        // Build atom_type_map and atoms
        atom::atom_type_map_t atom_type_map;
        gmp::containers::vector<atom::atom_flt> atoms;
        atoms.reserve(n_atoms);
        
        uint32_t next_type_id = 0;
        for (size_t i = 0; i < n_atoms; ++i) {
            std::string type_str = atom_types[i];
            if (atom_type_map.find(type_str) == atom_type_map.end()) {
                atom_type_map[type_str] = next_type_id++;
            }
            
            double occ = 1.0;
            if (atom_occupancies.size() > 0) {
                auto occ_buf = atom_occupancies.unchecked<1>();
                if (occ_buf.shape(0) != n_atoms) {
                    throw std::runtime_error("atom_occupancies must have same length as atom_positions");
                }
                occ = occ_buf(i);
            }
            
            atoms.push_back(atom::atom_flt{
                gmp::geometry::point3d_t<gmp::gmp_float>{
                    static_cast<gmp::gmp_float>(pos_buf(i, 0)),
                    static_cast<gmp::gmp_float>(pos_buf(i, 1)),
                    static_cast<gmp::gmp_float>(pos_buf(i, 2))
                },
                static_cast<gmp::gmp_float>(occ),
                atom_type_map[type_str]
            });
        }
        
        // Create unit_cell using C++ function
        return input::create_unit_cell_from_data(
            cell_len, cell_ang, atoms, atom_type_map,
            gmp::math::array3d_bool{periodicity[0], periodicity[1], periodicity[2]}
        );
    }

    // ============================================================================
    // Initialization Functions
    // ============================================================================
    
    // Initialize featurizer from JSON file
    std::unique_ptr<FeaturizerContext> initialize_featurizer_from_json(const std::string& json_file) {
        try {
            // Reset error state
            gmp::update_error(gmp::error_t::success);
            
            // Parse input
            std::unique_ptr<input::input_t> input = std::make_unique<input::input_t>(json_file);
            GMP_CHECK(get_last_error());
            
            // Create and return context
            return std::make_unique<FeaturizerContext>(std::move(input));
        } catch (const std::exception& e) {
            throw std::runtime_error("Featurizer initialization error: " + std::string(e.what()));
        }
    }
    
    // Initialize featurizer from direct parameters
    std::unique_ptr<FeaturizerContext> initialize_featurizer_from_params(
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
        bool enable_gpu = true,
        bool output_raw_data = false
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
            input->descriptor_config->set_output_raw_data(output_raw_data);
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
            
            // Create and return context
            return std::make_unique<FeaturizerContext>(std::move(input));
        } catch (const std::exception& e) {
            throw std::runtime_error("Featurizer initialization error: " + std::string(e.what()));
        }
    }
    
    // Initialize featurizer from direct data (no CIF file needed)
    std::unique_ptr<FeaturizerContext> initialize_featurizer_from_data(
        // Cell parameters
        py::array_t<double> cell_lengths,      // [a, b, c] in Angstroms
        py::array_t<double> cell_angles,       // [alpha, beta, gamma] in degrees
        // Atom data
        py::array_t<double> atom_positions,    // [n_atoms, 3] fractional coordinates
        const std::vector<std::string>& atom_types,   // [n_atoms] element symbols (Python list)
        py::array_t<double> atom_occupancies,  // [n_atoms] optional, defaults to 1.0
        // PSP file (still needed for now)
        const std::string& psp_file,
        // Descriptor configuration
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
        bool enable_gpu = true,
        bool output_raw_data = false,
        const std::array<bool, 3>& periodicity = {true, true, true}
    ) {
        try {
            // Reset error state
            gmp::update_error(gmp::error_t::success);
            
            // Create unit_cell from direct data using helper function
            auto unit_cell = create_unit_cell_from_data(
                cell_lengths, cell_angles, atom_positions, atom_types, 
                atom_occupancies, periodicity
            );
            GMP_CHECK(get_last_error());
            
            // Create input object with default constructor
            std::unique_ptr<input::input_t> input = std::make_unique<input::input_t>();
            GMP_CHECK(get_last_error());
            
            // Set PSP file (needed for psp_config)
            input->files->set_psp_file(psp_file);
            input->files->set_output_file(output_file);
            
            // Set descriptor configuration (same as initialize_featurizer_from_params)
            input->descriptor_config->set_square(square);
            input->descriptor_config->set_output_raw_data(output_raw_data);
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
            
            // Set reference positions if provided
            if (reference_grid.size() > 0) {
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
            
            for (double sigma : sigmas) {
                sigmas_vec.push_back(static_cast<gmp::gmp_float>(sigma));
            }
            
            for (const auto& pair : feature_lists) {
                feature_list_vec.emplace_back(pair.first, static_cast<gmp::gmp_float>(pair.second));
            }
            
            if (orders_vec.empty() && sigmas_vec.empty() && feature_list_vec.empty()) {
                orders_vec = {0, 1, 2, 3};
                sigmas_vec = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
            }
            
            input->descriptor_config->set_feature_list(orders_vec, sigmas_vec, feature_list_vec);
            
            // Create psp_config using separate function
            auto psp_config = input::create_psp_config(psp_file, unit_cell.get());
            GMP_CHECK(get_last_error());
            
            // Create ref_positions using separate function
            auto ref_positions = input::create_ref_positions(input.get(), unit_cell.get());
            GMP_CHECK(get_last_error());
            
            // Build custom init_result
            input::featurizer_init_result_t init_result;
            init_result.unit_cell = std::move(unit_cell);
            init_result.psp_config = std::move(psp_config);
            init_result.ref_positions = std::move(ref_positions);
            
            // Create context with custom initialization
            return std::make_unique<FeaturizerContext>(std::move(input), std::move(init_result));
        } catch (const std::exception& e) {
            throw std::runtime_error("Featurizer initialization error: " + std::string(e.what()));
        }
    }
    
    // ============================================================================
    // Computation Functions
    // ============================================================================
    
    // Compute features using CPU
    py::array_t<gmp::gmp_float> compute_features_cpu(FeaturizerContext& context) {
        try {
            // Reset error state
            gmp::update_error(gmp::error_t::success);
            
            ensure_cpu_initialized();
            
            // Call CPU featurizer computation
            auto result = gmp::run_cpu_featurizer_computation(
                context.get_descriptor_config(),
                context.get_unit_cell(),
                context.get_psp_config(),
                context.get_ref_positions()
            );
            GMP_CHECK(get_last_error());
            
            // Convert to numpy array
            return vector2d_to_numpy(result);
        } catch (const std::exception& e) {
            throw std::runtime_error("CPU featurizer computation error: " + std::string(e.what()));
        }
    }
    
    // Compute features using GPU
    py::array_t<gmp::gmp_float> compute_features_gpu(FeaturizerContext& context) {
        try {
            // Reset error state
            gmp::update_error(gmp::error_t::success);
            
#ifdef GMP_ENABLE_CUDA
            ensure_gpu_initialized();
            
            // Call GPU featurizer computation
            auto result = gmp::run_gpu_featurizer_computation(
                context.get_descriptor_config(),
                context.get_unit_cell(),
                context.get_psp_config(),
                context.get_ref_positions()
            );
            GMP_CHECK(get_last_error());
            
            // Get dimensions for numpy conversion
            const auto& ref_positions = context.get_ref_positions();
            size_t n_features = context.get_descriptor_config()->get_feature_list().size();
            size_t n_positions = ref_positions.size();
            
            // Handle different output formats based on output mode
            if (context.get_output_raw_data()) {
                // Raw data mode: output is in [position][raw_features] format
                size_t total_raw_features = 0;
                for (const auto& feature : context.get_descriptor_config()->get_feature_list()) {
                    int order = feature.order;
                    total_raw_features += (order < 0 ? 1 : gmp::mcsh::num_mcsh_values[order]);
                }
                return vector1d_to_numpy_raw(result, n_positions, total_raw_features);
            } else {
                // Normal mode: output is in [feature][position] format, need transpose
                return vector1d_to_numpy(result, n_positions, n_features);
            }
#else
            throw std::runtime_error("GPU support not compiled in this build");
#endif
        } catch (const std::exception& e) {
            throw std::runtime_error("GPU featurizer computation error: " + std::string(e.what()));
        }
    }
    
    // ============================================================================
    // Legacy Functions (for backward compatibility)
    // ============================================================================
    
    // Function that takes JSON file path and returns numpy array
    py::array_t<gmp::gmp_float> compute_features_from_json(const std::string& json_file) {
        try {
            // Initialize context
            auto context = initialize_featurizer_from_json(json_file);
            
            // Compute features based on GPU preference
            if (context->is_gpu_enabled()) {
                return compute_features_gpu(*context);
            } else {
                return compute_features_cpu(*context);
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
        bool enable_gpu = true,
        bool output_raw_data = false
    ) {
        try {
            // Initialize context
            auto context = initialize_featurizer_from_params(
                atom_file, psp_file, output_file, orders, sigmas, feature_lists,
                square, overlap_threshold, scaling_mode, uniform_reference_grid,
                reference_grid, num_bits_per_dim, num_threads, enable_gpu, output_raw_data
            );
            
            // Compute features based on GPU preference
            if (context->is_gpu_enabled()) {
                return compute_features_gpu(*context);
            } else {
                return compute_features_cpu(*context);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Featurizer error: " + std::string(e.what()));
        }
    }


    // Function to compute weighted square sum from raw data
    py::array_t<gmp::gmp_float> compute_weighted_square_sum(
        py::array_t<gmp::gmp_float> raw_data,
        const std::vector<int>& orders,
        bool square = false
    ) {
        try {
            // Reset error state
            gmp::update_error(gmp::error_t::success);
            
            // Get input array info
            auto buf = raw_data.unchecked<2>();
            size_t n_positions = buf.shape(0);
            size_t n_raw_features = buf.shape(1);
            
            // Verify orders list is not empty
            if (orders.empty()) {
                throw std::runtime_error("Orders list cannot be empty");
            }
            
            // Calculate expected total raw features and validate
            size_t expected_raw_features = 0;
            for (int order : orders) {
                if (order < -1 || order > 9) {
                    throw std::runtime_error("Invalid order: " + std::to_string(order) + ". Must be between -1 and 9.");
                }
                expected_raw_features += (order < 0 ? 1 : gmp::mcsh::num_mcsh_values[order]);
            }
            
            if (n_raw_features != expected_raw_features) {
                throw std::runtime_error(
                    "Mismatch between raw data features (" + std::to_string(n_raw_features) + 
                    ") and expected features (" + std::to_string(expected_raw_features) + 
                    ") based on orders list."
                );
            }
            
            // Create output array: [n_positions, n_features]
            size_t n_features = orders.size();
            auto result = py::array_t<gmp::gmp_float>(std::vector<size_t>{n_positions, n_features});
            auto result_buf = result.mutable_unchecked<2>();
            
            // Process each position
            for (size_t pos_idx = 0; pos_idx < n_positions; ++pos_idx) {
                size_t raw_offset = 0;
                
                // Process each feature
                for (size_t feat_idx = 0; feat_idx < n_features; ++feat_idx) {
                    int order = orders[feat_idx];
                    int num_values = (order < 0 ? 1 : gmp::mcsh::num_mcsh_values[order]);
                    
                    // Extract raw data for this feature
                    std::vector<gmp::gmp_float> desc_values(num_values);
                    for (int i = 0; i < num_values; ++i) {
                        desc_values[i] = static_cast<gmp::gmp_float>(buf(pos_idx, raw_offset + i));
                    }
                    
                    // Compute weighted square sum
                    gmp::gmp_float squareSum = gmp::math::weighted_square_sum(order, desc_values.data());
                    
                    // Store result (with or without square root)
                    if (square) {
                        result_buf(pos_idx, feat_idx) = squareSum;
                    } else {
                        result_buf(pos_idx, feat_idx) = std::sqrt(squareSum);
                    }
                    
                    raw_offset += num_values;
                }
            }
            
            return result;
        } catch (const std::exception& e) {
            throw std::runtime_error("Weighted square sum computation error: " + std::string(e.what()));
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
    
    // ============================================================================
    // FeaturizerContext class binding
    // ============================================================================
    py::class_<gmp::python::FeaturizerContext, std::unique_ptr<gmp::python::FeaturizerContext>>(m, "FeaturizerContext")
        .def("is_gpu_enabled", &gmp::python::FeaturizerContext::is_gpu_enabled,
             "Check if GPU is enabled for this context")
        .def("get_output_raw_data", &gmp::python::FeaturizerContext::get_output_raw_data,
             "Get output raw data flag");
    
    // ============================================================================
    // Initialization Functions
    // ============================================================================
    m.def("initialize_featurizer_from_json", &gmp::python::initialize_featurizer_from_json,
          "Initialize featurizer from JSON configuration file",
          py::arg("json_file"),
          py::return_value_policy::take_ownership);
    
    m.def("initialize_featurizer_from_params", &gmp::python::initialize_featurizer_from_params,
          "Initialize featurizer from direct parameters",
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
          py::arg("enable_gpu") = true,
          py::arg("output_raw_data") = false,
          py::return_value_policy::take_ownership);
    
    m.def("initialize_featurizer_from_data", &gmp::python::initialize_featurizer_from_data,
          "Initialize featurizer from direct data (no CIF file needed)",
          py::arg("cell_lengths"),
          py::arg("cell_angles"),
          py::arg("atom_positions"),
          py::arg("atom_types"),
          py::arg("atom_occupancies") = py::array_t<double>(),
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
          py::arg("enable_gpu") = true,
          py::arg("output_raw_data") = false,
          py::arg("periodicity") = std::array<bool, 3>{true, true, true},
          py::return_value_policy::take_ownership);
    
    // ============================================================================
    // Computation Functions
    // ============================================================================
    m.def("compute_features_cpu", &gmp::python::compute_features_cpu,
          "Compute features using CPU (requires initialized FeaturizerContext)",
          py::arg("context"));
    
    m.def("compute_features_gpu", &gmp::python::compute_features_gpu,
          "Compute features using GPU (requires initialized FeaturizerContext)",
          py::arg("context"));
    
    // ============================================================================
    // Legacy Functions (for backward compatibility)
    // ============================================================================
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
          py::arg("enable_gpu") = true,
          py::arg("output_raw_data") = false);
    
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
    
    // Weighted square sum computation from raw data
    m.def("compute_weighted_square_sum", &gmp::python::compute_weighted_square_sum,
          "Compute weighted square sum from raw descriptor data",
          py::arg("raw_data"),
          py::arg("orders"),
          py::arg("square") = false);
    
    // Add some constants
    m.attr("__version__") = "1.0.0";
    m.attr("__auto_cleanup__") = true;
    m.attr("__auto_init__") = true;
}
