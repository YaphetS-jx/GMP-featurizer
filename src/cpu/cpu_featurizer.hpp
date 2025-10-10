#pragma once
#include "input.hpp"
#include <vector>

namespace gmp {

    // Single function wrapper for CPU featurization
    std::vector<std::vector<gmp::gmp_float>> run_cpu_featurizer(input::input_t* input);

} // namespace gmp
