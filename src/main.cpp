#include <chrono>
#include "cpu_featurizer.hpp"
#include "input.hpp"
#include "util.hpp"
#include "atom.hpp"
#ifdef GMP_ENABLE_CUDA
#include "resources.hpp"
#include "gpu_featurizer.hpp"
#endif

int main(int argc, char* argv[]) {
    using namespace gmp;

    if (argc != 2) {
        std::cout << "ERROR: Please provide the path to the JSON file." << std::endl;
        return 1;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();

    // parse arguments
    std::unique_ptr<input::input_t> input = std::make_unique<input::input_t>(argv[1]);
    GMP_CHECK(get_last_error());    
    
#ifndef GMP_ENABLE_CUDA
    // CPU featurizer (CUDA not available)
    auto result = gmp::run_cpu_featurizer(input.get());
    gmp::util::write_vector_2d(result, input->files->get_output_file());
#else
    // Parse input to check GPU preference when CUDA is available
    bool use_gpu = input->get_descriptor_config()->get_enable_gpu();

    if (use_gpu) {
        // GPU featurizer
        auto result = gmp::run_gpu_featurizer(input.get());
        // Get reference positions for dimensions
        auto unit_cell = gmp::atom::unit_cell_flt(input->files->get_atom_file());
        auto ref_positions = gmp::atom::set_ref_positions(input->descriptor_config->get_ref_grid(), 
                                                         unit_cell.get_atoms());
        gmp::util::write_vector_1d(result, input->files->get_output_file(), 
                                   input->descriptor_config->get_feature_list().size(), 
                                   ref_positions.size(), false);
    } else {
        // CPU featurizer
        auto result = gmp::run_cpu_featurizer(input.get());
        gmp::util::write_vector_2d(result, input->files->get_output_file());
    }
#endif

    auto end_time = std::chrono::high_resolution_clock::now();
    auto walltime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken: " << static_cast<double>(walltime.count()) / 1000.0 << " seconds" << std::endl;

#ifdef GMP_ENABLE_CUDA
    gmp::resources::gmp_resource::instance().cleanup();
#endif
    
    return 0;
}
