#include <chrono>
#include "cpu_featurizer.hpp"
#include "input.hpp"
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
    gmp::run_cpu_featurizer(input.get());
#else
    // Parse input to check GPU preference when CUDA is available
    bool use_gpu = input->get_descriptor_config()->get_enable_gpu();

    if (use_gpu) {
        // GPU featurizer
        gmp::run_gpu_featurizer(input.get());
    } else {
        // CPU featurizer
        gmp::run_cpu_featurizer(input.get());
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
