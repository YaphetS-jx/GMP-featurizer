#pragma once
#include <cstdint>
#include "containers.hpp"
#include "math.hpp"
#include "gpu_qualifiers.hpp"

namespace gmp { namespace tree {

    using gmp::containers::vector;
    using gmp::math::array3d_int32;

    template <typename MortonCodeType = std::uint32_t, typename IndexType = std::int32_t>
    GPU_HOST_DEVICE
    struct internal_node_t {
        IndexType left, right;
        MortonCodeType lower_bound, upper_bound;
        internal_node_t(IndexType left, IndexType right, MortonCodeType lower_bound, MortonCodeType upper_bound)
            : left(left), right(right), lower_bound(lower_bound), upper_bound(upper_bound) {}
    };

    template <typename MortonCodeType, typename VecType = vector<array3d_int32>>
    GPU_HOST_DEVICE
    class compare_op_t {
    public: 
        virtual bool operator()(MortonCodeType query_lower_bound, MortonCodeType query_upper_bound) const = 0;
        virtual VecType operator()(MortonCodeType morton_code) const = 0;
    };
}}