#include <memory>
#include <chrono>
#include <iostream>
#include "input.hpp"
#include "error.hpp"
#include "atom.hpp"
#include "geometry.hpp"
#include "containers.hpp"
#include "featurizer.hpp"
#include "util.hpp"
#include "cpu_featurizer.hpp"

namespace gmp {

    // CPU featurization with raw data (core computation)
    std::vector<std::vector<gmp::gmp_float>> run_cpu_featurizer_computation(
        const input::descriptor_config_flt* descriptor_config,
        const atom::unit_cell_flt* unit_cell,
        const atom::psp_config_flt* psp_config,
        const containers::vector<geometry::point3d_t<gmp_float>>& ref_positions
    ) 
    {
        using namespace gmp;
        using namespace gmp::containers;

        // create featurizer_t
        std::unique_ptr<featurizer::featurizer_flt> featurizer = std::make_unique<featurizer::featurizer_flt>(
            descriptor_config, unit_cell, psp_config);
        GMP_CHECK(get_last_error());

        // compute features
        auto t1 = std::chrono::high_resolution_clock::now();
        auto result = featurizer->compute(ref_positions, descriptor_config, unit_cell, psp_config);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << "CPU featurizer time: " << static_cast<double>(compute_time.count()) << " ms" << std::endl;
        
        return result;
    }

    // CPU featurization from input_t (convenience wrapper)
    std::vector<std::vector<gmp::gmp_float>> run_cpu_featurizer(input::input_t* input) 
    {
        using namespace gmp;
        using namespace gmp::containers;
        std::cout << "Running CPU featurizer..." << std::endl;

        // Initialize featurizer (creates unit_cell, psp_config, and ref_positions)
        auto init_result = input::initialize_featurizer(input);
        GMP_CHECK(get_last_error());

        // Call the computation function with raw data
        return run_cpu_featurizer_computation(
            input->descriptor_config.get(),
            init_result.unit_cell.get(),
            init_result.psp_config.get(),
            init_result.ref_positions
        );
    }

} // namespace gmp
