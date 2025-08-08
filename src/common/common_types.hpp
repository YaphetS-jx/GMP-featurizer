#pragma once
#include <cstdint>
#include "containers.hpp"
#include "math.hpp"
#include "gpu_qualifiers.hpp"

namespace gmp { namespace tree {

    using gmp::containers::vector;
    using gmp::math::array3d_int32;

    template <typename MortonCodeType = std::uint32_t, typename IndexType = std::int32_t>
    struct internal_node_t {
        IndexType left, right;
        MortonCodeType lower_bound, upper_bound;
    };
}}