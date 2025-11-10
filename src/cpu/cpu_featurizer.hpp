#pragma once
#include "input.hpp"
#include "atom.hpp"
#include "containers.hpp"
#include "geometry.hpp"
#include <vector>

namespace gmp {

    // CPU featurization with raw data (core computation)
    std::vector<std::vector<gmp::gmp_float>> run_cpu_featurizer_computation(
        const input::descriptor_config_flt* descriptor_config,
        const atom::unit_cell_flt* unit_cell,
        const atom::psp_config_flt* psp_config,
        const containers::vector<geometry::point3d_t<gmp_float>>& ref_positions
    );

    // CPU featurization from input_t (convenience wrapper)
    std::vector<std::vector<gmp::gmp_float>> run_cpu_featurizer(input::input_t* input);

} // namespace gmp
