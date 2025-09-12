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

    void run_gpu_featurizer(input::input_t* input) 
    {
        using namespace gmp;
        using namespace gmp::containers;

        std::cout << "Running GPU featurizer..." << std::endl;

        // Ensure CUDA device is set before creating RMM resources
        cudaError_t cuda_status = cudaSetDevice(0);
        if (cuda_status != cudaSuccess) {
            std::cerr << "ERROR: Failed to set CUDA device: " << cudaGetErrorString(cuda_status) << std::endl;
            return;
        }

        // create unit cell
        std::unique_ptr<atom::unit_cell_flt> unit_cell = std::make_unique<atom::unit_cell_flt>(input->files->get_atom_file());
        GMP_CHECK(get_last_error());

        // create psp configuration
        std::unique_ptr<atom::psp_config_flt> psp_config = std::make_unique<atom::psp_config_flt>(input->files->get_psp_file(), unit_cell.get());
        GMP_CHECK(get_last_error());

        // create reference positions
        auto ref_positions = atom::set_ref_positions(input->descriptor_config->get_ref_grid(), unit_cell->get_atoms());

        // create featurizer_t
        std::unique_ptr<featurizer::featurizer_flt> featurizer = std::make_unique<featurizer::featurizer_flt>(
            input->descriptor_config.get(), unit_cell.get(), psp_config.get());
        GMP_CHECK(get_last_error());

        cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream();

        // compute features
        std::unique_ptr<featurizer::cuda_featurizer_flt> cuda_featurizer = std::make_unique<featurizer::cuda_featurizer_flt>(
            ref_positions, unit_cell->get_atoms(), psp_config.get(), 
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
            unit_cell->get_lattice(), stream
        );

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "GPU featurizer time: " << milliseconds << " ms" << std::endl;

        GMP_CHECK(get_last_error());
        util::write_vector_1d(result, input->files->get_output_file(), input->descriptor_config->get_feature_list().size(), ref_positions.size(), false);
        GMP_CHECK(get_last_error());
    }

} // namespace gmp
