#pragma once
#include "math.hpp"
#include "containers.hpp"
#include "mcsh_kernel.hpp"
#include "gpu_qualifiers.hpp"
#include "cuda_group_add.hpp"

namespace gmp { namespace mcsh {

    using gmp::math::array3d_t;
    using gmp::containers::vector;
    using gmp::math::weighted_square_sum;
    using gmp::group_add::HashTable;

    template <int Order, typename T> 
    GPU_HOST_DEVICE 
    void solid_mcsh(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* value, 
        const int base_key = 0, HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        if constexpr (Order <= 0) {
            mcsh::solid_mcsh_0(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 1) {
            mcsh::solid_mcsh_1(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 2) {
            mcsh::solid_mcsh_2(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 3) {
            mcsh::solid_mcsh_3(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 4) {
            mcsh::solid_mcsh_4(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 5) {
            mcsh::solid_mcsh_5(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 6) {
            mcsh::solid_mcsh_6(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 7) {
            mcsh::solid_mcsh_7(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 8) {
            mcsh::solid_mcsh_8(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else if constexpr (Order == 9) {
            mcsh::solid_mcsh_9(dr, r_sqr, temp, lambda, gamma, value, base_key, hash_table, local_tid);
        } else {
            static_assert(Order <= 9, "Unsupported MCSH order");
        }
    }

    constexpr int num_mcsh_values[] = {1, 3, 6, 10, 15, 21, 28, 36, 45};
}}