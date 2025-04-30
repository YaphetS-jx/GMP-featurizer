#pragma once
#include <vector>
#include <string>
#include "resources.hpp"
#include "boost_pool.hpp"

namespace gmp {

    // data structures
    template <typename T>
    using vec = std::vector<T, gmp::resources::pool_allocator<T>>;

    template <typename T>
    using gmp_unique_ptr = gmp::resources::pool_unique_ptr<T, gmp::resources::PoolType>;

    // error handling
    enum class error_t {
        success,
        matrix_singular,
    };
}