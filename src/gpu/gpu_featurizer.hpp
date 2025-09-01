#pragma once
#include "input.hpp"

namespace gmp {

    // Single function wrapper for GPU featurization
    void run_gpu_featurizer(input::input_t* input);

} // namespace gmp
