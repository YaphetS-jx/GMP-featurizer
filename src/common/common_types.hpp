#pragma once
#include <cstdint>
#include "containers.hpp"
#include "math.hpp"
#include "gpu_qualifiers.hpp"
#include "gmp_float.hpp"

namespace gmp { namespace tree {

    using gmp::containers::vector;
    using gmp::math::array3d_int32;
    using gmp::math::array3d_t;

    template <
        typename MortonCodeType = std::uint32_t,
        typename IndexType = std::int32_t,
        typename FloatType = gmp::gmp_float
    >
    struct internal_node_t {
        IndexType left, right;
        MortonCodeType lower_bound, upper_bound;
        array3d_t<FloatType> min_bounds;
        array3d_t<FloatType> max_bounds;
    };
}}