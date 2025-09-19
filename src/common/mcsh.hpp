#pragma once
#include "math.hpp"
#include "containers.hpp"
#include "mcsh_kernel.hpp"
#include "gpu_qualifiers.hpp"

namespace gmp { namespace mcsh {

    using gmp::math::array3d_t;
    using gmp::containers::vector;
    using gmp::math::weighted_square_sum;

    template <int Order, typename T> 
    GPU_HOST_DEVICE 
    void cuda_solid_mcsh(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* value) 
    {
        if constexpr (Order <= 0) {
            mcsh::solid_mcsh_0(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 1) {
            mcsh::solid_mcsh_1(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 2) {
            mcsh::solid_mcsh_2(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 3) {
            mcsh::solid_mcsh_3(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 4) {
            mcsh::solid_mcsh_4(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 5) {
            mcsh::solid_mcsh_5(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 6) {
            mcsh::solid_mcsh_6(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 7) {
            mcsh::solid_mcsh_7(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 8) {
            mcsh::solid_mcsh_8(dr, r_sqr, temp, lambda, gamma, value);
        } else if constexpr (Order == 9) {
            mcsh::solid_mcsh_9(dr, r_sqr, temp, lambda, gamma, value);
        } else {
            static_assert(Order <= 9, "Unsupported MCSH order");
        }
    }

    template <typename T> 
    GPU_HOST_DEVICE 
    void solid_mcsh(const int order, const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* value) {
        // Use switch statement inside kernel
        switch(order) {
            case -1: solid_mcsh_0(dr, r_sqr, temp, lambda, gamma, value); break;
            case 0: solid_mcsh_0(dr, r_sqr, temp, lambda, gamma, value); break;
            case 1: solid_mcsh_1(dr, r_sqr, temp, lambda, gamma, value); break;
            case 2: solid_mcsh_2(dr, r_sqr, temp, lambda, gamma, value); break;
            case 3: solid_mcsh_3(dr, r_sqr, temp, lambda, gamma, value); break;
            case 4: solid_mcsh_4(dr, r_sqr, temp, lambda, gamma, value); break;
            case 5: solid_mcsh_5(dr, r_sqr, temp, lambda, gamma, value); break;
            case 6: solid_mcsh_6(dr, r_sqr, temp, lambda, gamma, value); break;
            case 7: solid_mcsh_7(dr, r_sqr, temp, lambda, gamma, value); break;
            case 8: solid_mcsh_8(dr, r_sqr, temp, lambda, gamma, value); break;
            case 9: solid_mcsh_9(dr, r_sqr, temp, lambda, gamma, value); break;
            default: break;
        }
    }

    constexpr int num_mcsh_values[] = {1, 3, 6, 10, 15, 21, 28, 36, 45};
}}