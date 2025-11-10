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

    // GPU featurization with raw data (core computation)
    vector<gmp::gmp_float> run_gpu_featurizer_computation(
        const input::descriptor_config_flt* descriptor_config,
        const atom::unit_cell_flt* unit_cell,
        const atom::psp_config_flt* psp_config,
        const containers::vector<geometry::point3d_t<gmp_float>>& ref_positions
    ) 
    {
        using namespace gmp;

        // create featurizer_t
        std::unique_ptr<featurizer::featurizer_flt> featurizer = std::make_unique<featurizer::featurizer_flt>(
            descriptor_config, unit_cell, psp_config);
        GMP_CHECK(get_last_error());

        cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream();

        // compute features
        std::unique_ptr<featurizer::cuda_featurizer_flt> cuda_featurizer = std::make_unique<featurizer::cuda_featurizer_flt>(
            ref_positions, unit_cell->get_atoms(), psp_config, 
            featurizer->kernel_params_table_.get(), featurizer->cutoff_list_.get(), 
            featurizer->region_query_->get_unique_morton_codes(), 
            descriptor_config->get_num_bits_per_dim(), 
            featurizer->region_query_->get_offsets(), featurizer->region_query_->get_sorted_indexes());
        GMP_CHECK(get_last_error());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);

        // compute features
        auto result = cuda_featurizer->compute(
            descriptor_config->get_feature_list(), 
            descriptor_config->get_square(), 
            unit_cell->get_lattice(), stream,
            descriptor_config->get_output_raw_data()
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

    // GPU featurization from input_t (convenience wrapper)
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

        // Initialize featurizer (creates unit_cell, psp_config, and ref_positions)
        auto init_result = input::initialize_featurizer(input);
        GMP_CHECK(get_last_error());

        // Call the computation function with raw data
        return run_gpu_featurizer_computation(
            input->descriptor_config.get(),
            init_result.unit_cell.get(),
            init_result.psp_config.get(),
            init_result.ref_positions
        );
    }

} // namespace gmp
