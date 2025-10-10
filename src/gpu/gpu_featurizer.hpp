#pragma once
#include "input.hpp"
#include "containers.hpp"

namespace gmp {
    using containers::vector;

    // Single function wrapper for GPU featurization
    vector<gmp::gmp_float> run_gpu_featurizer(input::input_t* input);

} // namespace gmp
