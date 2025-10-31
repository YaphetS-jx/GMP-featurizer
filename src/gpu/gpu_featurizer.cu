#include <memory>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "input.hpp"
#include "error.hpp"
#include "atom.hpp"
#include "geometry.hpp"
#include "containers.hpp"
#include "featurizer.hpp"
#include "cuda_featurizer.hpp"
#include "util.hpp"
#include "gpu_featurizer.hpp"

namespace gmp {

    vector<gmp::gmp_float> run_gpu_featurizer(input::input_t* input) 
    {
        using namespace gmp;
        std::cout << "Running GPU featurizer..." << std::endl;

        // Ensure CUDA device is set before creating RMM resources
        cudaError_t cuda_status = cudaSetDevice(0);
        if (cuda_status != cudaSuccess) {
            std::cerr << "ERROR: Failed to set CUDA device: " << cudaGetErrorString(cuda_status) << std::endl;
            return vector<gmp::gmp_float>();
        }

        // create unit cell
        std::unique_ptr<atom::unit_cell_flt> unit_cell = std::make_unique<atom::unit_cell_flt>(input->files->get_atom_file());
        GMP_CHECK(get_last_error());

        // create psp configuration
        std::unique_ptr<atom::psp_config_flt> psp_config = std::make_unique<atom::psp_config_flt>(input->files->get_psp_file(), unit_cell.get());
        GMP_CHECK(get_last_error());

        // create reference positions
        gmp::containers::vector<gmp::geometry::point3d_t<gmp_float>> ref_positions;
        
        // Check if reference positions are provided directly
        if (input->has_reference_positions()) {
            // Use provided reference positions directly (no copying needed)
            ref_positions = input->get_reference_positions();
        } else if (!input->files->get_reference_grid_file().empty()) {
            // Read reference positions from file
            ref_positions = input->read_reference_grid_from_file(input->files->get_reference_grid_file());
            GMP_CHECK(get_last_error());
        } else {
            // Use original atom::set_ref_positions logic
            auto ref_grid = input->descriptor_config->get_ref_grid();
            ref_positions = atom::set_ref_positions(ref_grid, unit_cell->get_atoms());
        }

        // create featurizer_t
        std::unique_ptr<featurizer::featurizer_flt> featurizer = std::make_unique<featurizer::featurizer_flt>(
            input->descriptor_config.get(), unit_cell.get(), psp_config.get());
        GMP_CHECK(get_last_error());

        cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream();

        // compute features
        // Convert std::vector to gmp::containers::vector
        gmp::containers::vector<gmp::geometry::point3d_t<gmp_float>> ref_positions_container(ref_positions.begin(), ref_positions.end());
        std::unique_ptr<featurizer::cuda_featurizer_flt> cuda_featurizer = std::make_unique<featurizer::cuda_featurizer_flt>(
            ref_positions_container, unit_cell->get_atoms(), psp_config.get(), 
            featurizer->kernel_params_table_.get(), featurizer->cutoff_list_.get(), 
            featurizer->region_query_->get_unique_morton_codes(), 
            input->descriptor_config->get_num_bits_per_dim(), 
            featurizer->region_query_->get_offsets(), featurizer->region_query_->get_sorted_indexes());
        GMP_CHECK(get_last_error());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        // compute features
        auto result = cuda_featurizer->compute(
            input->descriptor_config->get_feature_list(), 
            input->descriptor_config->get_square(), 
            unit_cell->get_lattice(), stream,
            input->descriptor_config->get_output_raw_data()
        );

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "GPU featurizer time: " << milliseconds << " ms" << std::endl;

        // Check for CUDA errors before cleanup
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess) {
            std::cerr << "WARNING: CUDA error detected: " << cudaGetErrorString(cuda_error) << std::endl;
        }

        // Clean up CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        // Final CUDA synchronization
        cudaStreamSynchronize(stream);
        
        GMP_CHECK(get_last_error());
        
        return result;
    }

} // namespace gmp
