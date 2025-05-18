#pragma once
#include "boost_pool.hpp"
#include <vector>

namespace gmp {
    // Common type aliases for memory management
    template <typename T>
    using vec = std::vector<T, resources::pool_allocator<T>>;

    template <typename T>
    using gmp_unique_ptr = resources::pool_unique_ptr<T, resources::PoolType>;
} 