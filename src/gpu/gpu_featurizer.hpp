#pragma once
#include "input.hpp"
#include "containers.hpp"
#include "atom.hpp"
#include "geometry.hpp"

namespace gmp {
    using containers::vector;

    // GPU featurization with raw data (core computation)
    vector<gmp::gmp_float> run_gpu_featurizer_computation(
        const input::descriptor_config_flt* descriptor_config,
        const atom::unit_cell_flt* unit_cell,
        const atom::psp_config_flt* psp_config,
        const containers::vector<geometry::point3d_t<gmp_float>>& ref_positions
    );

    // GPU featurization from input_t (convenience wrapper)
    vector<gmp::gmp_float> run_gpu_featurizer(input::input_t* input);

} // namespace gmp
