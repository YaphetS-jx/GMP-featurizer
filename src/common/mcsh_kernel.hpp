#pragma once
#include "math.hpp"
#include "gpu_qualifiers.hpp"
#include "cuda_group_add.hpp"
#ifdef GMP_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace gmp { namespace mcsh {

    using gmp::math::array3d_t;
    using namespace gmp::group_add;

    template <class T>
    GPU_HOST_DEVICE
    inline void add_to_atomic(T* dst, const int key, const T x, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) {
    #ifndef __CUDA_ARCH__
        dst[key] += x;                 // host compilation: plain +=
    #else
        add_to_hash_table(dst, key, x, hash_table, local_tid);         // device compilation: atomic
    #endif
    }

    // Optimized polynomial calculations using constexpr
    template <typename T>
    GPU_HOST_DEVICE constexpr T P1(const T lambda_x0) {
        return lambda_x0;
    }
    
    template <typename T>
    GPU_HOST_DEVICE constexpr T P2(const T lambda_x0_2, const T inv_gamma) {        
        return (0.5 * inv_gamma) + lambda_x0_2;
    }
    
    template <typename T>
    GPU_HOST_DEVICE constexpr T P3(const T lambda_x0, const T lambda_x0_3, 
                                   const T inv_gamma) {
        return (1.5 * lambda_x0 * inv_gamma) + lambda_x0_3;
    }
    
    template <typename T>
    GPU_HOST_DEVICE constexpr T P4(const T lambda_x0_2, const T lambda_x0_4,
                                   const T inv_gamma, const T inv_gamma_2) {
        return 0.75 * inv_gamma_2 + 3.0 * lambda_x0_2 * inv_gamma + lambda_x0_4;
    }
    
    template <typename T>
    GPU_HOST_DEVICE constexpr T P5(const T lambda_x0, const T lambda_x0_3,
                                   const T lambda_x0_5, const T inv_gamma,
                                   const T inv_gamma_2) {
        return 3.75 * lambda_x0 * inv_gamma_2 + 5.0 * lambda_x0_3 * inv_gamma + lambda_x0_5;
    }
    
    template <typename T>
    GPU_HOST_DEVICE constexpr T P6(const T lambda_x0_2, const T lambda_x0_4,
                                   const T lambda_x0_6, const T inv_gamma,
                                   const T inv_gamma_2, const T inv_gamma_3) {
        return 1.875 * inv_gamma_3 + 11.25 * lambda_x0_2 * inv_gamma_2 + 7.5 * lambda_x0_4 * inv_gamma + lambda_x0_6;
    }
    
    template <typename T>
    GPU_HOST_DEVICE constexpr T P7(const T lambda_x0, const T lambda_x0_3,
                                   const T lambda_x0_5, const T lambda_x0_7,
                                   const T inv_gamma, const T inv_gamma_2,
                                   const T inv_gamma_3) {
        return 13.125 * lambda_x0 * inv_gamma_3 + 26.25 * lambda_x0_3 * inv_gamma_2 + 10.5 * lambda_x0_5 * inv_gamma + lambda_x0_7;
    }
    
    template <typename T>
    GPU_HOST_DEVICE constexpr T P8(const T lambda_x0_2, const T lambda_x0_4, 
                                   const T lambda_x0_6, const T lambda_x0_8,
                                   const T inv_gamma, const T inv_gamma_2, 
                                   const T inv_gamma_3, const T inv_gamma_4) {
        return 6.5625 * inv_gamma_4 + 52.5 * lambda_x0_2 * inv_gamma_3 + 52.5 * lambda_x0_4 * inv_gamma_2 + 14.0 * lambda_x0_6 * inv_gamma + lambda_x0_8;   
    }
    
    template <typename T>
    GPU_HOST_DEVICE constexpr T P9(const T lambda_x0, const T lambda_x0_3, 
                                   const T lambda_x0_5, const T lambda_x0_7,
                                   const T lambda_x0_9, const T inv_gamma, 
                                   const T inv_gamma_2, const T inv_gamma_3,
                                   const T inv_gamma_4) {
        return 59.0625 * lambda_x0 * inv_gamma_4 + 157.5 * lambda_x0_3 * inv_gamma_3 + 94.5 * lambda_x0_5 * inv_gamma_2 + 18.0 * lambda_x0_7 * inv_gamma + lambda_x0_9;
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_0(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        add_to_atomic(value, base_key, temp, hash_table, local_tid);
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_1(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const T temp_lambda = temp * lambda;
        #pragma unroll 3
        for (int dim = 0; dim < 3; ++dim) {
            add_to_atomic(value, base_key + dim, temp_lambda * dr[dim], hash_table, local_tid);
        }
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_2(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const array3d_t<T> lambda_dr = dr * lambda;
        const array3d_t<T> lambda_dr2 = lambda_dr * lambda_dr;
        const T inv_gamma = 1.0 / gamma;
        array3d_t<T> P2_vals;
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            P2_vals[dim] = P2(lambda_dr2[dim], inv_gamma);
        }

        const T sum_P2 = P2_vals[0] + P2_vals[1] + P2_vals[2];
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const T term = (3.0 * P2_vals[dim]) - sum_P2;
            add_to_atomic(value, base_key + dim, temp * term, hash_table, local_tid);
        }

        const T temp_times_three = 3.0 * temp;
        array3d_t<T> lambda_products;
        lambda_products[0] = lambda_dr[0] * lambda_dr[1];
        lambda_products[1] = lambda_dr[0] * lambda_dr[2];
        lambda_products[2] = lambda_dr[1] * lambda_dr[2];

        #pragma unroll
        for (int pair = 0; pair < 3; ++pair) {
            add_to_atomic(value, base_key + 3 + pair, temp_times_three * lambda_products[pair], hash_table, local_tid);
        }
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_3(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const array3d_t<T> lambda_dr = dr * lambda;
        const array3d_t<T> lambda_dr2 = lambda_dr * lambda_dr;
        const array3d_t<T> lambda_dr3 = lambda_dr2 * lambda_dr;
        const T inv_gamma = 1.0 / gamma;

        const array3d_t<T> P1_vals = lambda_dr;
        array3d_t<T> P2_vals;
        array3d_t<T> P3_vals;

        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            P2_vals[dim] = P2(lambda_dr2[dim], inv_gamma);
            P3_vals[dim] = P3(lambda_dr[dim], lambda_dr3[dim], inv_gamma);
        }

        const T sum_P2 = P2_vals[0] + P2_vals[1] + P2_vals[2];
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const T gp1_term = (6.0 * P3_vals[dim]) - (9.0 * P1_vals[dim] * (sum_P2 - P2_vals[dim]));
            add_to_atomic(value, base_key + dim, temp * gp1_term, hash_table, local_tid);
        }

        constexpr int pairs[6][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp2_term = (12.0 * P2_vals[a] * P1_vals[b]) - (3.0 * P3_vals[b]) - (3.0 * P1_vals[b] * P2_vals[c]);
            add_to_atomic(value, base_key + 3 + pair, temp * gp2_term, hash_table, local_tid);
        }

        const T gp3_term = 15.0 * temp * (lambda_dr[0] * lambda_dr[1] * lambda_dr[2]);
        add_to_atomic(value, base_key + 9, gp3_term, hash_table, local_tid);
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_4(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {

        const T lambda_x0 = lambda * dr[0];
        const T lambda_x0_2 = lambda_x0   * lambda_x0;
        const T lambda_x0_3 = lambda_x0_2 * lambda_x0;
        const T lambda_x0_4 = lambda_x0_3 * lambda_x0;
        const T lambda_y0 = lambda * dr[1];
        const T lambda_y0_2 = lambda_y0   * lambda_y0;
        const T lambda_y0_3 = lambda_y0_2 * lambda_y0;
        const T lambda_y0_4 = lambda_y0_3 * lambda_y0;
        const T lambda_z0 = lambda * dr[2];
        const T lambda_z0_2 = lambda_z0   * lambda_z0;
        const T lambda_z0_3 = lambda_z0_2 * lambda_z0;
        const T lambda_z0_4 = lambda_z0_3 * lambda_z0;

        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;

        const T P1x = lambda_x0;
        const T P1y = lambda_y0;
        const T P1z = lambda_z0;

        const T P2x = P2(lambda_x0_2, inv_gamma);
        const T P2y = P2(lambda_y0_2, inv_gamma);
        const T P2z = P2(lambda_z0_2, inv_gamma);

        const T P3x = P3(lambda_x0, lambda_x0_3, inv_gamma);
        const T P3y = P3(lambda_y0, lambda_y0_3, inv_gamma);
        const T P3z = P3(lambda_z0, lambda_z0_3, inv_gamma);

        const T P4x = P4(lambda_x0_2, lambda_x0_4, inv_gamma, inv_gamma_2);
        const T P4y = P4(lambda_y0_2, lambda_y0_4, inv_gamma, inv_gamma_2);
        const T P4z = P4(lambda_z0_2, lambda_z0_4, inv_gamma, inv_gamma_2);

        // group 1
        const T gp1_term_x = (24.0 * P4x) + (9.0 * (P4y + P4z)) - (72.0 * P2x * (P2y + P2z)) + (18.0 * P2y * P2z);
        const T gp1_term_y = (24.0 * P4y) + (9.0 * (P4x + P4z)) - (72.0 * P2y * (P2x + P2z)) + (18.0 * P2x * P2z);
        const T gp1_term_z = (24.0 * P4z) + (9.0 * (P4x + P4y)) - (72.0 * P2z * (P2x + P2y)) + (18.0 * P2x * P2y);

        const T gp1_miu_1 = temp * gp1_term_x;
        const T gp1_miu_2 = temp * gp1_term_y;
        const T gp1_miu_3 = temp * gp1_term_z;

        // add_to_atomic(value, gp1_miu_1, hash_table, local_tid);
        // add_to_atomic(value + 1, gp1_miu_2, hash_table, local_tid);
        // add_to_atomic(value + 2, gp1_miu_3, hash_table, local_tid);
        add_to_atomic(value, base_key, gp1_miu_1, hash_table, local_tid);
        add_to_atomic(value, base_key + 1, gp1_miu_2, hash_table, local_tid);
        add_to_atomic(value, base_key + 2, gp1_miu_3, hash_table, local_tid);

        // group 2
        const T gp2_term_1 = (60.0 * P3x * P1y) - (45.0 * (P1x * P3y + P1x * P1y * P2z));
        const T gp2_term_2 = (60.0 * P3y * P1x) - (45.0 * (P1y * P3x + P1y * P1x * P2z));
        const T gp2_term_3 = (60.0 * P3x * P1z) - (45.0 * (P1x * P3z + P1x * P1z * P2y));
        const T gp2_term_4 = (60.0 * P3z * P1x) - (45.0 * (P1z * P3x + P1z * P1x * P2y));
        const T gp2_term_5 = (60.0 * P3y * P1z) - (45.0 * (P1y * P3z + P1y * P1z * P2x));
        const T gp2_term_6 = (60.0 * P3z * P1y) - (45.0 * (P1z * P3y + P1z * P1y * P2x));

        const T gp2_miu_1 = temp * gp2_term_1;
        const T gp2_miu_2 = temp * gp2_term_2;
        const T gp2_miu_3 = temp * gp2_term_3;
        const T gp2_miu_4 = temp * gp2_term_4;
        const T gp2_miu_5 = temp * gp2_term_5;
        const T gp2_miu_6 = temp * gp2_term_6;

        // add_to_atomic(value + 3, gp2_miu_1, hash_table, local_tid);
        // add_to_atomic(value + 4, gp2_miu_2, hash_table, local_tid);
        // add_to_atomic(value + 5, gp2_miu_3, hash_table, local_tid);
        // add_to_atomic(value + 6, gp2_miu_4, hash_table, local_tid);
        // add_to_atomic(value + 7, gp2_miu_5, hash_table, local_tid);
        // add_to_atomic(value + 8, gp2_miu_6, hash_table, local_tid);
        add_to_atomic(value, base_key + 3, gp2_miu_1, hash_table, local_tid);
        add_to_atomic(value, base_key + 4, gp2_miu_2, hash_table, local_tid);
        add_to_atomic(value, base_key + 5, gp2_miu_3, hash_table, local_tid);
        add_to_atomic(value, base_key + 6, gp2_miu_4, hash_table, local_tid);
        add_to_atomic(value, base_key + 7, gp2_miu_5, hash_table, local_tid);
        add_to_atomic(value, base_key + 8, gp2_miu_6, hash_table, local_tid);
        


        // group 3
        const T gp3_term_1 = (81.0 * P2x * P2y) - (12.0 * (P4x + P4y)) + (3.0 * P4z) - (9.0 * P2z * (P2x + P2y));
        const T gp3_term_2 = (81.0 * P2x * P2z) - (12.0 * (P4x + P4z)) + (3.0 * P4y) - (9.0 * P2y * (P2x + P2z));
        const T gp3_term_3 = (81.0 * P2y * P2z) - (12.0 * (P4y + P4z)) + (3.0 * P4x) - (9.0 * P2x * (P2y + P2z));

        const T gp3_miu_1 = temp * gp3_term_1;
        const T gp3_miu_2 = temp * gp3_term_2;
        const T gp3_miu_3 = temp * gp3_term_3;

        // add_to_atomic(value + 9, gp3_miu_1, hash_table, local_tid);
        // add_to_atomic(value + 10, gp3_miu_2, hash_table, local_tid);
        // add_to_atomic(value + 11, gp3_miu_3, hash_table, local_tid);
        add_to_atomic(value, base_key + 9, gp3_miu_1, hash_table, local_tid);
        add_to_atomic(value, base_key + 10, gp3_miu_2, hash_table, local_tid);
        add_to_atomic(value, base_key + 11, gp3_miu_3, hash_table, local_tid);

        // group 4
        const T gp4_term_1 = 90.0 * P2x * P1y * P1z - 15.0 * (P3y * P1z + P1y * P3z);
        const T gp4_term_2 = 90.0 * P2y * P1x * P1z - 15.0 * (P3x * P1z + P1x * P3z);
        const T gp4_term_3 = 90.0 * P2z * P1x * P1y - 15.0 * (P3x * P1y + P1x * P3y);

        const T gp4_miu_1 = temp * gp4_term_1;
        const T gp4_miu_2 = temp * gp4_term_2;
        const T gp4_miu_3 = temp * gp4_term_3;

        // add_to_atomic(value + 12, gp4_miu_1, hash_table, local_tid);
        // add_to_atomic(value + 13, gp4_miu_2, hash_table, local_tid);
        // add_to_atomic(value + 14, gp4_miu_3, hash_table, local_tid);
        add_to_atomic(value, base_key + 12, gp4_miu_1, hash_table, local_tid);
        add_to_atomic(value, base_key + 13, gp4_miu_2, hash_table, local_tid);
        add_to_atomic(value, base_key + 14, gp4_miu_3, hash_table, local_tid);
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_5(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {

        const T lambda_x0 = lambda * dr[0];
        const T lambda_x0_2 = lambda_x0 * lambda_x0;
        const T lambda_x0_3 = lambda_x0_2 * lambda_x0;
        const T lambda_x0_4 = lambda_x0_3 * lambda_x0;
        const T lambda_x0_5 = lambda_x0_4 * lambda_x0;
        const T lambda_y0 = lambda * dr[1];
        const T lambda_y0_2 = lambda_y0 * lambda_y0;
        const T lambda_y0_3 = lambda_y0_2 * lambda_y0;
        const T lambda_y0_4 = lambda_y0_3 * lambda_y0;
        const T lambda_y0_5 = lambda_y0_4 * lambda_y0;
        const T lambda_z0 = lambda * dr[2];
        const T lambda_z0_2 = lambda_z0 * lambda_z0;
        const T lambda_z0_3 = lambda_z0_2 * lambda_z0;
        const T lambda_z0_4 = lambda_z0_3 * lambda_z0;
        const T lambda_z0_5 = lambda_z0_4 * lambda_z0;

        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;

        const T P1x = lambda_x0;
        const T P1y = lambda_y0;
        const T P1z = lambda_z0;

        const T P2x = P2(lambda_x0_2, inv_gamma);
        const T P2y = P2(lambda_y0_2, inv_gamma);
        const T P2z = P2(lambda_z0_2, inv_gamma);

        const T P3x = P3(lambda_x0, lambda_x0_3, inv_gamma);
        const T P3y = P3(lambda_y0, lambda_y0_3, inv_gamma);
        const T P3z = P3(lambda_z0, lambda_z0_3, inv_gamma);

        const T P4x = P4(lambda_x0_2, lambda_x0_4, inv_gamma, inv_gamma_2);
        const T P4y = P4(lambda_y0_2, lambda_y0_4, inv_gamma, inv_gamma_2);
        const T P4z = P4(lambda_z0_2, lambda_z0_4, inv_gamma, inv_gamma_2);

        const T P5x = P5(lambda_x0, lambda_x0_3, lambda_x0_5, inv_gamma, inv_gamma_2);
        const T P5y = P5(lambda_y0, lambda_y0_3, lambda_y0_5, inv_gamma, inv_gamma_2);
        const T P5z = P5(lambda_z0, lambda_z0_3, lambda_z0_5, inv_gamma, inv_gamma_2);

        // group 1
        const T gp1_term_x = (120.0 * P5x) - (600.0 * P3x * (P2y + P2z)) + (225.0 * P1x * (P4y + P4z)) + (450.0 * P1x * P2y * P2z);
        const T gp1_term_y = (120.0 * P5y) - (600.0 * P3y * (P2x + P2z)) + (225.0 * P1y * (P4x + P4z)) + (450.0 * P1y * P2x * P2z);
        const T gp1_term_z = (120.0 * P5z) - (600.0 * P3z * (P2x + P2y)) + (225.0 * P1z * (P4x + P4y)) + (450.0 * P1z * P2x * P2y);

        const T gp1_miu_1 = temp * gp1_term_x;
        const T gp1_miu_2 = temp * gp1_term_y;
        const T gp1_miu_3 = temp * gp1_term_z;

        value[0] += gp1_miu_1;
        value[1] += gp1_miu_2;
        value[2] += gp1_miu_3;

        // group 2 
        const T gp2_term_1 = (360.0 * P4x * P1y) - (540.0 * P2x * (P3y + P1y * P2z)) + (45.0 * (P5y + P1y * P4z)) + (90.0 * P3y * P2z);
        const T gp2_term_2 = (360.0 * P4y * P1x) - (540.0 * P2y * (P3x + P1x * P2z)) + (45.0 * (P5x + P1x * P4z)) + (90.0 * P3x * P2z);
        const T gp2_term_3 = (360.0 * P4x * P1z) - (540.0 * P2x * (P3z + P1z * P2y)) + (45.0 * (P5z + P1z * P4y)) + (90.0 * P3z * P2y);
        const T gp2_term_4 = (360.0 * P4z * P1x) - (540.0 * P2z * (P3x + P1x * P2y)) + (45.0 * (P5x + P1x * P4y)) + (90.0 * P3x * P2y);
        const T gp2_term_5 = (360.0 * P4y * P1z) - (540.0 * P2y * (P3z + P1z * P2x)) + (45.0 * (P5z + P1z * P4x)) + (90.0 * P3z * P2x);
        const T gp2_term_6 = (360.0 * P4z * P1y) - (540.0 * P2z * (P3y + P1y * P2x)) + (45.0 * (P5y + P1y * P4x)) + (90.0 * P3y * P2x);

        const T gp2_miu_1 = temp * gp2_term_1;
        const T gp2_miu_2 = temp * gp2_term_2;
        const T gp2_miu_3 = temp * gp2_term_3;
        const T gp2_miu_4 = temp * gp2_term_4;
        const T gp2_miu_5 = temp * gp2_term_5;
        const T gp2_miu_6 = temp * gp2_term_6;

        value[3] += gp2_miu_1;
        value[4] += gp2_miu_2;
        value[5] += gp2_miu_3;
        value[6] += gp2_miu_4;
        value[7] += gp2_miu_5;
        value[8] += gp2_miu_6;


        // group 3
        const T gp3_term_1 = (615.0 * P3x * P2y) - (60.0 * P5x) - (15.0 * P3x * P2z) - (270.0 * P1x * P4y) + (45.0 * P1x * P4z) - (225.0 * P1x * P2y * P2z);
        const T gp3_term_2 = (615.0 * P3y * P2x) - (60.0 * P5y) - (15.0 * P3y * P2z) - (270.0 * P1y * P4x) + (45.0 * P1y * P4z) - (225.0 * P1y * P2x * P2z);
        const T gp3_term_3 = (615.0 * P3x * P2z) - (60.0 * P5x) - (15.0 * P3x * P2y) - (270.0 * P1x * P4z) + (45.0 * P1x * P4y) - (225.0 * P1x * P2z * P2y);
        const T gp3_term_4 = (615.0 * P3z * P2x) - (60.0 * P5z) - (15.0 * P3z * P2y) - (270.0 * P1z * P4x) + (45.0 * P1z * P4y) - (225.0 * P1z * P2x * P2y);
        const T gp3_term_5 = (615.0 * P3y * P2z) - (60.0 * P5y) - (15.0 * P3y * P2x) - (270.0 * P1y * P4z) + (45.0 * P1y * P4x) - (225.0 * P1y * P2z * P2x);
        const T gp3_term_6 = (615.0 * P3z * P2y) - (60.0 * P5z) - (15.0 * P3z * P2x) - (270.0 * P1z * P4y) + (45.0 * P1z * P4x) - (225.0 * P1z * P2y * P2x);

        const T gp3_miu_1 = temp * gp3_term_1;
        const T gp3_miu_2 = temp * gp3_term_2;
        const T gp3_miu_3 = temp * gp3_term_3;
        const T gp3_miu_4 = temp * gp3_term_4;
        const T gp3_miu_5 = temp * gp3_term_5;
        const T gp3_miu_6 = temp * gp3_term_6;

        value[9] += gp3_miu_1;
        value[10] += gp3_miu_2;
        value[11] += gp3_miu_3;
        value[12] += gp3_miu_4;
        value[13] += gp3_miu_5;
        value[14] += gp3_miu_6;


        // group 4
        const T gp4_term_1 = (630.0 * P3x * P1y * P1z) - (315 * P1x * (P3y * P1z + P1y * P3z));
        const T gp4_term_2 = (630.0 * P3y * P1x * P1z) - (315 * P1y * (P3x * P1z + P1x * P3z));
        const T gp4_term_3 = (630.0 * P3z * P1x * P1y) - (315 * P1z * (P3x * P1y + P1x * P3y));

        const T gp4_miu_1 = temp * gp4_term_1;
        const T gp4_miu_2 = temp * gp4_term_2;
        const T gp4_miu_3 = temp * gp4_term_3;

        value[15] += gp4_miu_1;
        value[16] += gp4_miu_2;
        value[17] += gp4_miu_3;


        // group 5
        const T gp5_term_1 = (765.0 * P2x * P2y * P1z) - (90.0 * P1z * (P4x + P4y)) - (75.0 * P3z * (P2x + P2y)) + (15.0 * P5z);
        const T gp5_term_2 = (765.0 * P2x * P2z * P1y) - (90.0 * P1y * (P4x + P4z)) - (75.0 * P3y * (P2x + P2z)) + (15.0 * P5y);
        const T gp5_term_3 = (765.0 * P2y * P2z * P1x) - (90.0 * P1x * (P4y + P4z)) - (75.0 * P3x * (P2y + P2z)) + (15.0 * P5x);

        const T gp5_miu_1 = temp * gp5_term_1;
        const T gp5_miu_2 = temp * gp5_term_2;
        const T gp5_miu_3 = temp * gp5_term_3;

        value[18] += gp5_miu_1;
        value[19] += gp5_miu_2;
        value[20] += gp5_miu_3;
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_6(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {

        const T lambda_x0 = lambda * dr[0];
        const T lambda_x0_2 = lambda_x0 * lambda_x0;
        const T lambda_x0_3 = lambda_x0_2 * lambda_x0;
        const T lambda_x0_4 = lambda_x0_3 * lambda_x0;
        const T lambda_x0_5 = lambda_x0_4 * lambda_x0;
        const T lambda_x0_6 = lambda_x0_5 * lambda_x0;
        const T lambda_y0 = lambda * dr[1];
        const T lambda_y0_2 = lambda_y0 * lambda_y0;
        const T lambda_y0_3 = lambda_y0_2 * lambda_y0;
        const T lambda_y0_4 = lambda_y0_3 * lambda_y0;
        const T lambda_y0_5 = lambda_y0_4 * lambda_y0;
        const T lambda_y0_6 = lambda_y0_5 * lambda_y0;
        const T lambda_z0 = lambda * dr[2];
        const T lambda_z0_2 = lambda_z0 * lambda_z0;
        const T lambda_z0_3 = lambda_z0_2 * lambda_z0;
        const T lambda_z0_4 = lambda_z0_3 * lambda_z0;
        const T lambda_z0_5 = lambda_z0_4 * lambda_z0;
        const T lambda_z0_6 = lambda_z0_5 * lambda_z0;

        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;
        const T inv_gamma_3 = inv_gamma_2 * inv_gamma;

        const T P1x = lambda_x0;
        const T P1y = lambda_y0;
        const T P1z = lambda_z0;

        const T P2x = P2(lambda_x0_2, inv_gamma);
        const T P2y = P2(lambda_y0_2, inv_gamma);
        const T P2z = P2(lambda_z0_2, inv_gamma);

        const T P3x = P3(lambda_x0, lambda_x0_3, inv_gamma);
        const T P3y = P3(lambda_y0, lambda_y0_3, inv_gamma);
        const T P3z = P3(lambda_z0, lambda_z0_3, inv_gamma);

        const T P4x = P4(lambda_x0_2, lambda_x0_4, inv_gamma, inv_gamma_2);
        const T P4y = P4(lambda_y0_2, lambda_y0_4, inv_gamma, inv_gamma_2);
        const T P4z = P4(lambda_z0_2, lambda_z0_4, inv_gamma, inv_gamma_2);

        const T P5x = P5(lambda_x0, lambda_x0_3, lambda_x0_5, inv_gamma, inv_gamma_2);
        const T P5y = P5(lambda_y0, lambda_y0_3, lambda_y0_5, inv_gamma, inv_gamma_2);
        const T P5z = P5(lambda_z0, lambda_z0_3, lambda_z0_5, inv_gamma, inv_gamma_2);

        const T P6x = P6(lambda_x0_2, lambda_x0_4, lambda_x0_6, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P6y = P6(lambda_y0_2, lambda_y0_4, lambda_y0_6, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P6z = P6(lambda_z0_2, lambda_z0_4, lambda_z0_6, inv_gamma, inv_gamma_2, inv_gamma_3);

        // group 1
        const T gp1_term_1 = (720.0 * P6x) - (5400.0 * P4x * P2y) - (5400.0 * P4x * P2z) + (4050.0 * P2x * P4y) + (8100.0 * P2x * P2y * P2z) + (4050.0 * P2x * P4z) - (225.0 * P6y) - (675.0 * P4y * P2z) - (675.0 * P2y * P4z) - (225.0 * P6z);
        const T gp1_term_2 = (720.0 * P6y) - (5400.0 * P4y * P2x) - (5400.0 * P4y * P2z) + (4050.0 * P2y * P4x) + (8100.0 * P2y * P2x * P2z) + (4050.0 * P2y * P4z) - (225.0 * P6x) - (675.0 * P4x * P2z) - (675.0 * P2x * P4z) - (225.0 * P6z);
        const T gp1_term_3 = (720.0 * P6z) - (5400.0 * P4z * P2x) - (5400.0 * P4z * P2y) + (4050.0 * P2z * P4x) + (8100.0 * P2z * P2x * P2y) + (4050.0 * P2z * P4y) - (225.0 * P6x) - (675.0 * P4x * P2y) - (675.0 * P2x * P4y) - (225.0 * P6y);

        const T gp1_miu_1 = temp * gp1_term_1;
        const T gp1_miu_2 = temp * gp1_term_2;
        const T gp1_miu_3 = temp * gp1_term_3;

        value[0] += gp1_miu_1;
        value[1] += gp1_miu_2;
        value[2] += gp1_miu_3;


        // group 2 
        const T gp2_term_1 = (2520.0 * P5x * P1y) - (6300.0 * P3x * P3y) - (6300.0 * P3x * P1y * P2z) + (1575.0 * P1x * P5y) + (3150.0 * P1x * P3y * P2z) + (1575.0 * P1x * P1y * P4z);
        const T gp2_term_2 = (2520.0 * P5y * P1x) - (6300.0 * P3y * P3x) - (6300.0 * P3y * P1x * P2z) + (1575.0 * P1y * P5x) + (3150.0 * P1y * P3x * P2z) + (1575.0 * P1y * P1x * P4z);
        const T gp2_term_3 = (2520.0 * P5x * P1z) - (6300.0 * P3x * P3z) - (6300.0 * P3x * P1z * P2y) + (1575.0 * P1x * P5z) + (3150.0 * P1x * P3z * P2y) + (1575.0 * P1x * P1z * P4y);
        const T gp2_term_4 = (2520.0 * P5z * P1x) - (6300.0 * P3z * P3x) - (6300.0 * P3z * P1x * P2y) + (1575.0 * P1z * P5x) + (3150.0 * P1z * P3x * P2y) + (1575.0 * P1z * P1x * P4y);
        const T gp2_term_5 = (2520.0 * P5y * P1z) - (6300.0 * P3y * P3z) - (6300.0 * P3y * P1z * P2x) + (1575.0 * P1y * P5z) + (3150.0 * P1y * P3z * P2x) + (1575.0 * P1y * P1z * P4x);
        const T gp2_term_6 = (2520.0 * P5z * P1y) - (6300.0 * P3z * P3y) - (6300.0 * P3z * P1y * P2x) + (1575.0 * P1z * P5y) + (3150.0 * P1z * P3y * P2x) + (1575.0 * P1z * P1y * P4x);

        const T gp2_miu_1 = temp * gp2_term_1;
        const T gp2_miu_2 = temp * gp2_term_2;
        const T gp2_miu_3 = temp * gp2_term_3;
        const T gp2_miu_4 = temp * gp2_term_4;
        const T gp2_miu_5 = temp * gp2_term_5;
        const T gp2_miu_6 = temp * gp2_term_6;

        value[3] += gp2_miu_1;
        value[4] += gp2_miu_2;
        value[5] += gp2_miu_3;
        value[6] += gp2_miu_4;
        value[7] += gp2_miu_5;
        value[8] += gp2_miu_6;

        // group 3 
        const T gp3_term_1 = (5220.0 * P4x * P2y) - (360.0 * P6x) + (180.0 * P4x * P2z) - (4545.0 * P2x * P4y) - (4050.0 * P2x * P2y * P2z) + (495.0 * P2x * P4z) + (270.0 * P6y) + (495.0 * P4y * P2z) + (180.0 * P2y * P4z) - (45.0 * P6z);
        const T gp3_term_2 = (5220.0 * P4y * P2x) - (360.0 * P6y) + (180.0 * P4y * P2z) - (4545.0 * P2y * P4x) - (4050.0 * P2y * P2x * P2z) + (495.0 * P2y * P4z) + (270.0 * P6x) + (495.0 * P4x * P2z) + (180.0 * P2x * P4z) - (45.0 * P6z);
        const T gp3_term_3 = (5220.0 * P4x * P2z) - (360.0 * P6x) + (180.0 * P4x * P2y) - (4545.0 * P2x * P4z) - (4050.0 * P2x * P2z * P2y) + (495.0 * P2x * P4y) + (270.0 * P6z) + (495.0 * P4z * P2y) + (180.0 * P2z * P4y) - (45.0 * P6y);
        const T gp3_term_4 = (5220.0 * P4z * P2x) - (360.0 * P6z) + (180.0 * P4z * P2y) - (4545.0 * P2z * P4x) - (4050.0 * P2z * P2x * P2y) + (495.0 * P2z * P4y) + (270.0 * P6x) + (495.0 * P4x * P2y) + (180.0 * P2x * P4y) - (45.0 * P6y);
        const T gp3_term_5 = (5220.0 * P4y * P2z) - (360.0 * P6y) + (180.0 * P4y * P2x) - (4545.0 * P2y * P4z) - (4050.0 * P2y * P2z * P2x) + (495.0 * P2y * P4x) + (270.0 * P6z) + (495.0 * P4z * P2x) + (180.0 * P2z * P4x) - (45.0 * P6x);
        const T gp3_term_6 = (5220.0 * P4z * P2y) - (360.0 * P6z) + (180.0 * P4z * P2x) - (4545.0 * P2z * P4y) - (4050.0 * P2z * P2y * P2x) + (495.0 * P2z * P4x) + (270.0 * P6y) + (495.0 * P4y * P2x) + (180.0 * P2y * P4x) - (45.0 * P6x);

        const T gp3_miu_1 = temp * gp3_term_1;
        const T gp3_miu_2 = temp * gp3_term_2;
        const T gp3_miu_3 = temp * gp3_term_3;
        const T gp3_miu_4 = temp * gp3_term_4;
        const T gp3_miu_5 = temp * gp3_term_5;
        const T gp3_miu_6 = temp * gp3_term_6;

        value[9] += gp3_miu_1;
        value[10] += gp3_miu_2;
        value[11] += gp3_miu_3;
        value[12] += gp3_miu_4;
        value[13] += gp3_miu_5;
        value[14] += gp3_miu_6;

        // group 4
        const T gp4_term_1 = (5040.0 * P4x * P1y * P1z) - (5040.0 * P2x * P3y * P1z) - (5040.0 * P2x * P1y * P3z) + (315.0 * P5y * P1z) + (630.0 * P3y * P3z) + (315.0 * P1y * P5z);
        const T gp4_term_2 = (5040.0 * P4y * P1x * P1z) - (5040.0 * P2y * P3x * P1z) - (5040.0 * P2y * P1x * P3z) + (315.0 * P5x * P1z) + (630.0 * P3x * P3z) + (315.0 * P1x * P5z);
        const T gp4_term_3 = (5040.0 * P4z * P1x * P1y) - (5040.0 * P2z * P3x * P1y) - (5040.0 * P2z * P1x * P3y) + (315.0 * P5x * P1y) + (630.0 * P3x * P3y) + (315.0 * P1x * P5y);

        const T gp4_miu_1 = temp * gp4_term_1;
        const T gp4_miu_2 = temp * gp4_term_2;
        const T gp4_miu_3 = temp * gp4_term_3;

        value[15] += gp4_miu_1;
        value[16] += gp4_miu_2;
        value[17] += gp4_miu_3;

        // group 5
        const T gp5_term_1 = (6615.0 * P3x * P3y) - (1890.0 * P5x * P1y) - (945.0 * P3x * P1y * P2z) - (1890.0 * P1x * P5y) - (945.0 * P1x * P3y * P2z) + (945.0 * P1x * P1y * P4z);
        const T gp5_term_2 = (6615.0 * P3x * P3z) - (1890.0 * P5x * P1z) - (945.0 * P3x * P1z * P2y) - (1890.0 * P1x * P5z) - (945.0 * P1x * P3z * P2y) + (945.0 * P1x * P1z * P4y);
        const T gp5_term_3 = (6615.0 * P3y * P3z) - (1890.0 * P5y * P1z) - (945.0 * P3y * P1z * P2x) - (1890.0 * P1y * P5z) - (945.0 * P1y * P3z * P2x) + (945.0 * P1y * P1z * P4x);

        const T gp5_miu_1 = temp * gp5_term_1;
        const T gp5_miu_2 = temp * gp5_term_2;
        const T gp5_miu_3 = temp * gp5_term_3;

        value[18] += gp5_miu_1;
        value[19] += gp5_miu_2;
        value[20] += gp5_miu_3;

        // group 6
        const T gp6_term_1 = (7245.0 * P3x * P2y * P1z) - (630.0 * P5x * P1z) - (315.0 * P3x * P3z) - (2520.0 * P1x * P4y * P1z) - (2205.0 * P1x * P2y * P3z) + (315.0 * P1x * P5z);
        const T gp6_term_2 = (7245.0 * P3y * P2x * P1z) - (630.0 * P5y * P1z) - (315.0 * P3y * P3z) - (2520.0 * P1y * P4x * P1z) - (2205.0 * P1y * P2x * P3z) + (315.0 * P1y * P5z);
        const T gp6_term_3 = (7245.0 * P3x * P2z * P1y) - (630.0 * P5x * P1y) - (315.0 * P3x * P3y) - (2520.0 * P1x * P4z * P1y) - (2205.0 * P1x * P2z * P3y) + (315.0 * P1x * P5y);
        const T gp6_term_4 = (7245.0 * P3z * P2x * P1y) - (630.0 * P5z * P1y) - (315.0 * P3z * P3y) - (2520.0 * P1z * P4x * P1y) - (2205.0 * P1z * P2x * P3y) + (315.0 * P1z * P5y);
        const T gp6_term_5 = (7245.0 * P3y * P2z * P1x) - (630.0 * P5y * P1x) - (315.0 * P3y * P3x) - (2520.0 * P1y * P4z * P1x) - (2205.0 * P1y * P2z * P3x) + (315.0 * P1y * P5x);
        const T gp6_term_6 = (7245.0 * P3z * P2y * P1x) - (630.0 * P5z * P1x) - (315.0 * P3z * P3x) - (2520.0 * P1z * P4y * P1x) - (2205.0 * P1z * P2y * P3x) + (315.0 * P1z * P5x);

        const T gp6_miu_1 = temp * gp6_term_1;
        const T gp6_miu_2 = temp * gp6_term_2;
        const T gp6_miu_3 = temp * gp6_term_3;
        const T gp6_miu_4 = temp * gp6_term_4;
        const T gp6_miu_5 = temp * gp6_term_5;
        const T gp6_miu_6 = temp * gp6_term_6;

        value[21] += gp6_miu_1;
        value[22] += gp6_miu_2;
        value[23] += gp6_miu_3;
        value[24] += gp6_miu_4;
        value[25] += gp6_miu_5;
        value[26] += gp6_miu_6;


        // group 7
        const T gp7_term = (8100.0 * P2x * P2y * P2z) - (675.0 * P4x * P2y) - (675.0 * P2x * P4y) - (675.0 * P4x * P2z) - (675.0 * P2x * P4z) - (675.0 * P4y * P2z) - (675.0 * P2y * P4z) + (90.0 * P6x) + (90.0 * P6y) + (90.0 * P6z);
        const T gp7_m = temp * gp7_term;
        value[27] += gp7_m;

    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_7(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {

        const T lambda_x0 = lambda * dr[0];
        const T lambda_x0_2 = lambda_x0 * lambda_x0;
        const T lambda_x0_3 = lambda_x0_2 * lambda_x0;
        const T lambda_x0_4 = lambda_x0_3 * lambda_x0;
        const T lambda_x0_5 = lambda_x0_4 * lambda_x0;
        const T lambda_x0_6 = lambda_x0_5 * lambda_x0;
        const T lambda_x0_7 = lambda_x0_6 * lambda_x0;

        const T lambda_y0 = lambda * dr[1];
        const T lambda_y0_2 = lambda_y0 * lambda_y0;
        const T lambda_y0_3 = lambda_y0_2 * lambda_y0;
        const T lambda_y0_4 = lambda_y0_3 * lambda_y0;
        const T lambda_y0_5 = lambda_y0_4 * lambda_y0;
        const T lambda_y0_6 = lambda_y0_5 * lambda_y0;
        const T lambda_y0_7 = lambda_y0_6 * lambda_y0;

        const T lambda_z0 = lambda * dr[2];
        const T lambda_z0_2 = lambda_z0 * lambda_z0;
        const T lambda_z0_3 = lambda_z0_2 * lambda_z0;
        const T lambda_z0_4 = lambda_z0_3 * lambda_z0;
        const T lambda_z0_5 = lambda_z0_4 * lambda_z0;
        const T lambda_z0_6 = lambda_z0_5 * lambda_z0;
        const T lambda_z0_7 = lambda_z0_6 * lambda_z0;

        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;
        const T inv_gamma_3 = inv_gamma_2 * inv_gamma;

        const T P1x = lambda_x0;
        const T P1y = lambda_y0;
        const T P1z = lambda_z0;

        const T P2x = P2(lambda_x0_2, inv_gamma);
        const T P2y = P2(lambda_y0_2, inv_gamma);
        const T P2z = P2(lambda_z0_2, inv_gamma);

        const T P3x = P3(lambda_x0, lambda_x0_3, inv_gamma);
        const T P3y = P3(lambda_y0, lambda_y0_3, inv_gamma);
        const T P3z = P3(lambda_z0, lambda_z0_3, inv_gamma);

        const T P4x = P4(lambda_x0_2, lambda_x0_4, inv_gamma, inv_gamma_2);
        const T P4y = P4(lambda_y0_2, lambda_y0_4, inv_gamma, inv_gamma_2);
        const T P4z = P4(lambda_z0_2, lambda_z0_4, inv_gamma, inv_gamma_2);

        const T P5x = P5(lambda_x0, lambda_x0_3, lambda_x0_5, inv_gamma, inv_gamma_2);
        const T P5y = P5(lambda_y0, lambda_y0_3, lambda_y0_5, inv_gamma, inv_gamma_2);
        const T P5z = P5(lambda_z0, lambda_z0_3, lambda_z0_5, inv_gamma, inv_gamma_2);

        const T P6x = P6(lambda_x0_2, lambda_x0_4, lambda_x0_6, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P6y = P6(lambda_y0_2, lambda_y0_4, lambda_y0_6, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P6z = P6(lambda_z0_2, lambda_z0_4, lambda_z0_6, inv_gamma, inv_gamma_2, inv_gamma_3);

        const T P7x = P7(lambda_x0, lambda_x0_3, lambda_x0_5, lambda_x0_7, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P7y = P7(lambda_y0, lambda_y0_3, lambda_y0_5, lambda_y0_7, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P7z = P7(lambda_z0, lambda_z0_3, lambda_z0_5, lambda_z0_7, inv_gamma, inv_gamma_2, inv_gamma_3);

        // group 1
        const T gp1_term_1 = (5040.0 * P7x) - (52920.0 * P5x * P2y) - (52920.0 * P5x * P2z) + (66150.0 * P3x * P4y) + (132300.0 * P3x * P2y * P2z) + (66150.0 * P3x * P4z) - (11025.0 * P1x * P6y) - (33075.0 * P1x * P4y * P2z) - (33075.0 * P1x * P2y * P4z) - (11025.0 * P1x * P6z);
        const T gp1_term_2 = (5040.0 * P7y) - (52920.0 * P5y * P2x) - (52920.0 * P5y * P2z) + (66150.0 * P3y * P4x) + (132300.0 * P3y * P2x * P2z) + (66150.0 * P3y * P4z) - (11025.0 * P1y * P6x) - (33075.0 * P1y * P4x * P2z) - (33075.0 * P1y * P2x * P4z) - (11025.0 * P1y * P6z);
        const T gp1_term_3 = (5040.0 * P7z) - (52920.0 * P5z * P2x) - (52920.0 * P5z * P2y) + (66150.0 * P3z * P4x) + (132300.0 * P3z * P2x * P2y) + (66150.0 * P3z * P4y) - (11025.0 * P1z * P6x) - (33075.0 * P1z * P4x * P2y) - (33075.0 * P1z * P2x * P4y) - (11025.0 * P1z * P6y);
        
        const T gp1_miu_1 = temp * gp1_term_1;
        const T gp1_miu_2 = temp * gp1_term_2;
        const T gp1_miu_3 = temp * gp1_term_3;

        value[0] += gp1_miu_1;
        value[1] += gp1_miu_2;
        value[2] += gp1_miu_3;

        // group 2
        const T gp2_term_1 = (20160.0 * P6x * P1y) - (75600.0 * P4x * P3y) - (75600.0 * P4x * P1y * P2z) + (37800.0 * P2x * P5y) + (75600.0 * P2x * P3y * P2z) + (37800.0 * P2x * P1y * P4z) - (1575.0 * P7y) - (4725.0 * P5y * P2z) - (4725.0 * P3y * P4z) - (1575.0 * P1y * P6z);
        const T gp2_term_2 = (20160.0 * P6y * P1x) - (75600.0 * P4y * P3x) - (75600.0 * P4y * P1x * P2z) + (37800.0 * P2y * P5x) + (75600.0 * P2y * P3x * P2z) + (37800.0 * P2y * P1x * P4z) - (1575.0 * P7x) - (4725.0 * P5x * P2z) - (4725.0 * P3x * P4z) - (1575.0 * P1x * P6z);
        const T gp2_term_3 = (20160.0 * P6x * P1z) - (75600.0 * P4x * P3z) - (75600.0 * P4x * P1z * P2y) + (37800.0 * P2x * P5z) + (75600.0 * P2x * P3z * P2y) + (37800.0 * P2x * P1z * P4y) - (1575.0 * P7z) - (4725.0 * P5z * P2y) - (4725.0 * P3z * P4y) - (1575.0 * P1z * P6y);
        const T gp2_term_4 = (20160.0 * P6z * P1x) - (75600.0 * P4z * P3x) - (75600.0 * P4z * P1x * P2y) + (37800.0 * P2z * P5x) + (75600.0 * P2z * P3x * P2y) + (37800.0 * P2z * P1x * P4y) - (1575.0 * P7x) - (4725.0 * P5x * P2y) - (4725.0 * P3x * P4y) - (1575.0 * P1x * P6y);
        const T gp2_term_5 = (20160.0 * P6y * P1z) - (75600.0 * P4y * P3z) - (75600.0 * P4y * P1z * P2x) + (37800.0 * P2y * P5z) + (75600.0 * P2y * P3z * P2x) + (37800.0 * P2y * P1z * P4x) - (1575.0 * P7z) - (4725.0 * P5z * P2x) - (4725.0 * P3z * P4x) - (1575.0 * P1z * P6x);
        const T gp2_term_6 = (20160.0 * P6z * P1y) - (75600.0 * P4z * P3y) - (75600.0 * P4z * P1y * P2x) + (37800.0 * P2z * P5y) + (75600.0 * P2z * P3y * P2x) + (37800.0 * P2z * P1y * P4x) - (1575.0 * P7y) - (4725.0 * P5y * P2x) - (4725.0 * P3y * P4x) - (1575.0 * P1y * P6x);
        
        const T gp2_miu_1 = temp * gp2_term_1;
        const T gp2_miu_2 = temp * gp2_term_2;
        const T gp2_miu_3 = temp * gp2_term_3;
        const T gp2_miu_4 = temp * gp2_term_4;
        const T gp2_miu_5 = temp * gp2_term_5;
        const T gp2_miu_6 = temp * gp2_term_6;

        value[3] += gp2_miu_1;
        value[4] += gp2_miu_2;
        value[5] += gp2_miu_3;
        value[6] += gp2_miu_4;
        value[7] += gp2_miu_5;
        value[8] += gp2_miu_6;

        // group 3
        const T gp3_term_1 = (49140.0 * P5x * P2y) - (2520.0 * P7x) + (3780.0 * P5x * P2z) - (70875.0 * P3x * P4y) - (66150.0 * P3x * P2y * P2z) + (4725.0 * P3x * P4z) + (12600.0 * P1x * P6y) + (23625.0 * P1x * P4y * P2z) + (9450.0 * P1x * P2y * P4z) - (1575.0 * P1x * P6z);
        const T gp3_term_2 = (49140.0 * P5y * P2x) - (2520.0 * P7y) + (3780.0 * P5y * P2z) - (70875.0 * P3y * P4x) - (66150.0 * P3y * P2x * P2z) + (4725.0 * P3y * P4z) + (12600.0 * P1y * P6x) + (23625.0 * P1y * P4x * P2z) + (9450.0 * P1y * P2x * P4z) - (1575.0 * P1y * P6z);
        const T gp3_term_3 = (49140.0 * P5x * P2z) - (2520.0 * P7x) + (3780.0 * P5x * P2y) - (70875.0 * P3x * P4z) - (66150.0 * P3x * P2z * P2y) + (4725.0 * P3x * P4y) + (12600.0 * P1x * P6z) + (23625.0 * P1x * P4z * P2y) + (9450.0 * P1x * P2z * P4y) - (1575.0 * P1x * P6y);
        const T gp3_term_4 = (49140.0 * P5z * P2x) - (2520.0 * P7z) + (3780.0 * P5z * P2y) - (70875.0 * P3z * P4x) - (66150.0 * P3z * P2x * P2y) + (4725.0 * P3z * P4y) + (12600.0 * P1z * P6x) + (23625.0 * P1z * P4x * P2y) + (9450.0 * P1z * P2x * P4y) - (1575.0 * P1z * P6y);
        const T gp3_term_5 = (49140.0 * P5y * P2z) - (2520.0 * P7y) + (3780.0 * P5y * P2x) - (70875.0 * P3y * P4z) - (66150.0 * P3y * P2z * P2x) + (4725.0 * P3y * P4x) + (12600.0 * P1y * P6z) + (23625.0 * P1y * P4z * P2x) + (9450.0 * P1y * P2z * P4x) - (1575.0 * P1y * P6x);
        const T gp3_term_6 = (49140.0 * P5z * P2y) - (2520.0 * P7z) + (3780.0 * P5z * P2x) - (70875.0 * P3z * P4y) - (66150.0 * P3z * P2y * P2x) + (4725.0 * P3z * P4x) + (12600.0 * P1z * P6y) + (23625.0 * P1z * P4y * P2x) + (9450.0 * P1z * P2y * P4x) - (1575.0 * P1z * P6x);
        
        const T gp3_miu_1 = temp * gp3_term_1;
        const T gp3_miu_2 = temp * gp3_term_2;
        const T gp3_miu_3 = temp * gp3_term_3;
        const T gp3_miu_4 = temp * gp3_term_4;
        const T gp3_miu_5 = temp * gp3_term_5;
        const T gp3_miu_6 = temp * gp3_term_6;

        value[9] += gp3_miu_1;
        value[10] += gp3_miu_2;
        value[11] += gp3_miu_3;
        value[12] += gp3_miu_4;
        value[13] += gp3_miu_5;
        value[14] += gp3_miu_6;


        // group 4
        const T gp4_term_1 = (45360.0 * P5x * P1y * P1z) - (75600.0 * P3x * P3y * P1z) - (75600.0 * P3x * P1y * P3z) + (14175.0 * P1x * P5y * P1z) + (28350.0 * P1x * P3y * P3z) + (14175.0 * P1x * P1y * P5z);
        const T gp4_term_2 = (45360.0 * P5y * P1x * P1z) - (75600.0 * P3y * P3x * P1z) - (75600.0 * P3y * P1x * P3z) + (14175.0 * P1y * P5x * P1z) + (28350.0 * P1y * P3x * P3z) + (14175.0 * P1y * P1x * P5z);
        const T gp4_term_3 = (45360.0 * P5z * P1x * P1y) - (75600.0 * P3z * P3x * P1y) - (75600.0 * P3z * P1x * P3y) + (14175.0 * P1z * P5x * P1y) + (28350.0 * P1z * P3x * P3y) + (14175.0 * P1z * P1x * P5y);
        
        const T gp4_miu_1 = temp * gp4_term_1;
        const T gp4_miu_2 = temp * gp4_term_2;
        const T gp4_miu_3 = temp * gp4_term_3;

        value[15] += gp4_miu_1;
        value[16] += gp4_miu_2;
        value[17] += gp4_miu_3;


        // group 5
        const T gp5_term_1 = (75600.0 * P4x * P3y) - (15120.0 * P6x * P1y) - (42525.0 * P2x * P5y) - (28350.0 * P2x * P3y * P2z) + (14175.0 * P2x * P1y * P4z) + (1890.0 * P7y) + (2835.0 * P5y * P2z) - (945.0 * P1y * P6z);
        const T gp5_term_2 = (75600.0 * P4y * P3x) - (15120.0 * P6y * P1x) - (42525.0 * P2y * P5x) - (28350.0 * P2y * P3x * P2z) + (14175.0 * P2y * P1x * P4z) + (1890.0 * P7x) + (2835.0 * P5x * P2z) - (945.0 * P1x * P6z);
        const T gp5_term_3 = (75600.0 * P4x * P3z) - (15120.0 * P6x * P1z) - (42525.0 * P2x * P5z) - (28350.0 * P2x * P3z * P2y) + (14175.0 * P2x * P1z * P4y) + (1890.0 * P7z) + (2835.0 * P5z * P2y) - (945.0 * P1z * P6y);
        const T gp5_term_4 = (75600.0 * P4z * P3x) - (15120.0 * P6z * P1x) - (42525.0 * P2z * P5x) - (28350.0 * P2z * P3x * P2y) + (14175.0 * P2z * P1x * P4y) + (1890.0 * P7x) + (2835.0 * P5x * P2y) - (945.0 * P1x * P6y);
        const T gp5_term_5 = (75600.0 * P4y * P3z) - (15120.0 * P6y * P1z) - (42525.0 * P2y * P5z) - (28350.0 * P2y * P3z * P2x) + (14175.0 * P2y * P1z * P4x) + (1890.0 * P7z) + (2835.0 * P5z * P2x) - (945.0 * P1z * P6x);
        const T gp5_term_6 = (75600.0 * P4z * P3y) - (15120.0 * P6z * P1y) - (42525.0 * P2z * P5y) - (28350.0 * P2z * P3y * P2x) + (14175.0 * P2z * P1y * P4x) + (1890.0 * P7y) + (2835.0 * P5y * P2x) - (945.0 * P1y * P6x);
        
        const T gp5_miu_1 = temp * gp5_term_1;
        const T gp5_miu_2 = temp * gp5_term_2;
        const T gp5_miu_3 = temp * gp5_term_3;
        const T gp5_miu_4 = temp * gp5_term_4;
        const T gp5_miu_5 = temp * gp5_term_5;
        const T gp5_miu_6 = temp * gp5_term_6;

        value[18] += gp5_miu_1;
        value[19] += gp5_miu_2;
        value[20] += gp5_miu_3;
        value[21] += gp5_miu_4;
        value[22] += gp5_miu_5;
        value[23] += gp5_miu_6;


        // group 6
        const T gp6_term_1 = (75600.0 * P4x * P2y * P1z) - (5040.0 * P6x * P1z) - (51975.0 * P2x * P4y * P1z) - (47250.0 * P2x * P2y * P3z) + (4725.0 * P2x * P5z) + (2520.0 * P6y * P1z) + (4725.0 * P4y * P3z) + (1890.0 * P2y * P5z) - (315.0 * P7z);
        const T gp6_term_2 = (75600.0 * P4y * P2x * P1z) - (5040.0 * P6y * P1z) - (51975.0 * P2y * P4x * P1z) - (47250.0 * P2y * P2x * P3z) + (4725.0 * P2y * P5z) + (2520.0 * P6x * P1z) + (4725.0 * P4x * P3z) + (1890.0 * P2x * P5z) - (315.0 * P7z);
        const T gp6_term_3 = (75600.0 * P4x * P2z * P1y) - (5040.0 * P6x * P1y) - (51975.0 * P2x * P4z * P1y) - (47250.0 * P2x * P2z * P3y) + (4725.0 * P2x * P5y) + (2520.0 * P6z * P1y) + (4725.0 * P4z * P3y) + (1890.0 * P2z * P5y) - (315.0 * P7y);
        const T gp6_term_4 = (75600.0 * P4z * P2x * P1y) - (5040.0 * P6z * P1y) - (51975.0 * P2z * P4x * P1y) - (47250.0 * P2z * P2x * P3y) + (4725.0 * P2z * P5y) + (2520.0 * P6x * P1y) + (4725.0 * P4x * P3y) + (1890.0 * P2x * P5y) - (315.0 * P7y);
        const T gp6_term_5 = (75600.0 * P4y * P2z * P1x) - (5040.0 * P6y * P1x) - (51975.0 * P2y * P4z * P1x) - (47250.0 * P2y * P2z * P3x) + (4725.0 * P2y * P5x) + (2520.0 * P6z * P1x) + (4725.0 * P4z * P3x) + (1890.0 * P2z * P5x) - (315.0 * P7x);
        const T gp6_term_6 = (75600.0 * P4z * P2y * P1x) - (5040.0 * P6z * P1x) - (51975.0 * P2z * P4y * P1x) - (47250.0 * P2z * P2y * P3x) + (4725.0 * P2z * P5x) + (2520.0 * P6y * P1x) + (4725.0 * P4y * P3x) + (1890.0 * P2y * P5x) - (315.0 * P7x);  
        
        const T gp6_miu_1 = temp * gp6_term_1;
        const T gp6_miu_2 = temp * gp6_term_2;
        const T gp6_miu_3 = temp * gp6_term_3;
        const T gp6_miu_4 = temp * gp6_term_4;
        const T gp6_miu_5 = temp * gp6_term_5;
        const T gp6_miu_6 = temp * gp6_term_6;

        value[24] += gp6_miu_1;
        value[25] += gp6_miu_2;
        value[26] += gp6_miu_3;
        value[27] += gp6_miu_4;
        value[28] += gp6_miu_5;
        value[29] += gp6_miu_6;


        // group 7
        const T gp7_term_1 = (89775.0 * P3x * P3y * P1z) - (22680.0 * P5x * P1y * P1z) - (14175.0 * P3x * P1y * P3z) - (22680.0 * P1x * P5y * P1z) - (14175.0 * P1x * P3y * P3z) + (8505.0 * P1x * P1y * P5z);
        const T gp7_term_2 = (89775.0 * P3x * P3z * P1y) - (22680.0 * P5x * P1z * P1y) - (14175.0 * P3x * P1z * P3y) - (22680.0 * P1x * P5z * P1y) - (14175.0 * P1x * P3z * P3y) + (8505.0 * P1x * P1z * P5y);
        const T gp7_term_3 = (89775.0 * P3y * P3z * P1x) - (22680.0 * P5y * P1z * P1x) - (14175.0 * P3y * P1z * P3x) - (22680.0 * P1y * P5z * P1x) - (14175.0 * P1y * P3z * P3x) + (8505.0 * P1y * P1z * P5x);
        
        const T gp7_miu_1 = temp * gp7_term_1;
        const T gp7_miu_2 = temp * gp7_term_2;
        const T gp7_miu_3 = temp * gp7_term_3;

        value[30] += gp7_miu_1;
        value[31] += gp7_miu_2;
        value[32] += gp7_miu_3;


        // group 8 
        const T gp8_term_1 = (94500.0 * P3x * P2y * P2z) - (6615.0 * P5x * P2z) - (4725.0 * P3x * P4z) - (6615.0 * P5x * P2y) - (4725.0 * P3x * P4y) + (630.0 * P7x) - (23625.0 * P1x * P4y * P2z) - (23625.0 * P1x * P2y * P4z) + (2520.0 * P1x * P6z) + (2520.0 * P1x * P6y);
        const T gp8_term_2 = (94500.0 * P3y * P2x * P2z) - (6615.0 * P5y * P2z) - (4725.0 * P3y * P4z) - (6615.0 * P5y * P2x) - (4725.0 * P3y * P4x) + (630.0 * P7y) - (23625.0 * P1y * P4x * P2z) - (23625.0 * P1y * P2x * P4z) + (2520.0 * P1y * P6z) + (2520.0 * P1y * P6x);
        const T gp8_term_3 = (94500.0 * P3z * P2x * P2y) - (6615.0 * P5z * P2y) - (4725.0 * P3z * P4y) - (6615.0 * P5z * P2x) - (4725.0 * P3z * P4x) + (630.0 * P7z) - (23625.0 * P1z * P4x * P2y) - (23625.0 * P1z * P2x * P4y) + (2520.0 * P1z * P6y) + (2520.0 * P1z * P6x);
        
        const T gp8_miu_1 = temp * gp8_term_1;
        const T gp8_miu_2 = temp * gp8_term_2;
        const T gp8_miu_3 = temp * gp8_term_3;

        value[33] += gp8_miu_1;
        value[34] += gp8_miu_2;
        value[35] += gp8_miu_3;

    }
    
    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_8(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const T lambda_x0 = lambda * dr[0];
        const T lambda_x0_2 = lambda_x0 * lambda_x0;
        const T lambda_x0_3 = lambda_x0_2 * lambda_x0;
        const T lambda_x0_4 = lambda_x0_3 * lambda_x0;
        const T lambda_x0_5 = lambda_x0_4 * lambda_x0;
        const T lambda_x0_6 = lambda_x0_5 * lambda_x0;
        const T lambda_x0_7 = lambda_x0_6 * lambda_x0;
        const T lambda_x0_8 = lambda_x0_7 * lambda_x0;

        const T lambda_y0 = lambda * dr[1];
        const T lambda_y0_2 = lambda_y0 * lambda_y0;
        const T lambda_y0_3 = lambda_y0_2 * lambda_y0;
        const T lambda_y0_4 = lambda_y0_3 * lambda_y0;
        const T lambda_y0_5 = lambda_y0_4 * lambda_y0;
        const T lambda_y0_6 = lambda_y0_5 * lambda_y0;
        const T lambda_y0_7 = lambda_y0_6 * lambda_y0;
        const T lambda_y0_8 = lambda_y0_7 * lambda_y0;

        const T lambda_z0 = lambda * dr[2];
        const T lambda_z0_2 = lambda_z0 * lambda_z0;
        const T lambda_z0_3 = lambda_z0_2 * lambda_z0;
        const T lambda_z0_4 = lambda_z0_3 * lambda_z0;
        const T lambda_z0_5 = lambda_z0_4 * lambda_z0;
        const T lambda_z0_6 = lambda_z0_5 * lambda_z0;
        const T lambda_z0_7 = lambda_z0_6 * lambda_z0;
        const T lambda_z0_8 = lambda_z0_7 * lambda_z0;

        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;
        const T inv_gamma_3 = inv_gamma_2 * inv_gamma;
        const T inv_gamma_4 = inv_gamma_3 * inv_gamma;

        const T P1x = lambda_x0;
        const T P1y = lambda_y0;
        const T P1z = lambda_z0;

        const T P2x = P2(lambda_x0_2, inv_gamma);
        const T P2y = P2(lambda_y0_2, inv_gamma);
        const T P2z = P2(lambda_z0_2, inv_gamma);

        const T P3x = P3(lambda_x0, lambda_x0_3, inv_gamma);
        const T P3y = P3(lambda_y0, lambda_y0_3, inv_gamma);
        const T P3z = P3(lambda_z0, lambda_z0_3, inv_gamma);

        const T P4x = P4(lambda_x0_2, lambda_x0_4, inv_gamma, inv_gamma_2);
        const T P4y = P4(lambda_y0_2, lambda_y0_4, inv_gamma, inv_gamma_2);
        const T P4z = P4(lambda_z0_2, lambda_z0_4, inv_gamma, inv_gamma_2);

        const T P5x = P5(lambda_x0, lambda_x0_3, lambda_x0_5, inv_gamma, inv_gamma_2);
        const T P5y = P5(lambda_y0, lambda_y0_3, lambda_y0_5, inv_gamma, inv_gamma_2);
        const T P5z = P5(lambda_z0, lambda_z0_3, lambda_z0_5, inv_gamma, inv_gamma_2);

        const T P6x = P6(lambda_x0_2, lambda_x0_4, lambda_x0_6, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P6y = P6(lambda_y0_2, lambda_y0_4, lambda_y0_6, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P6z = P6(lambda_z0_2, lambda_z0_4, lambda_z0_6, inv_gamma, inv_gamma_2, inv_gamma_3);

        const T P7x = P7(lambda_x0, lambda_x0_3, lambda_x0_5, lambda_x0_7, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P7y = P7(lambda_y0, lambda_y0_3, lambda_y0_5, lambda_y0_7, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P7z = P7(lambda_z0, lambda_z0_3, lambda_z0_5, lambda_z0_7, inv_gamma, inv_gamma_2, inv_gamma_3);

        const T P8x = P8(lambda_x0_2, lambda_x0_4, lambda_x0_6, lambda_x0_8, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
        const T P8y = P8(lambda_y0_2, lambda_y0_4, lambda_y0_6, lambda_y0_8, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
        const T P8z = P8(lambda_z0_2, lambda_z0_4, lambda_z0_6, lambda_z0_8, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);

        // group 1
        const T gp1_term_1 = (40320.0 * P8x) - (564480.0 * P6x * P2y) - (564480.0 * P6x * P2z) + (1058400.0 * P4x * P4y) + (2116800.0 * P4x * P2y * P2z) + (1058400.0 * P4x * P4z) - (352800.0 * P2x * P6y) - (1058400.0 * P2x * P4y * P2z) - (1058400.0 * P2x * P2y * P4z) - (352800.0 * P2x * P6z) + (11025.0 * P8y) + (44100.0 * P6y * P2z) + (66150.0 * P4y * P4z) + (44100.0 * P2y * P6z) + (11025.0 * P8z);
        const T gp1_term_2 = (40320.0 * P8y) - (564480.0 * P6y * P2x) - (564480.0 * P6y * P2z) + (1058400.0 * P4y * P4x) + (2116800.0 * P4y * P2x * P2z) + (1058400.0 * P4y * P4z) - (352800.0 * P2y * P6x) - (1058400.0 * P2y * P4x * P2z) - (1058400.0 * P2y * P2x * P4z) - (352800.0 * P2y * P6z) + (11025.0 * P8x) + (44100.0 * P6x * P2z) + (66150.0 * P4x * P4z) + (44100.0 * P2x * P6z) + (11025.0 * P8z);
        const T gp1_term_3 = (40320.0 * P8z) - (564480.0 * P6z * P2x) - (564480.0 * P6z * P2y) + (1058400.0 * P4z * P4x) + (2116800.0 * P4z * P2x * P2y) + (1058400.0 * P4z * P4y) - (352800.0 * P2z * P6x) - (1058400.0 * P2z * P4x * P2y) - (1058400.0 * P2z * P2x * P4y) - (352800.0 * P2z * P6y) + (11025.0 * P8x) + (44100.0 * P6x * P2y) + (66150.0 * P4x * P4y) + (44100.0 * P2x * P6y) + (11025.0 * P8y);

        const T gp1_miu_1 = temp * gp1_term_1;
        const T gp1_miu_2 = temp * gp1_term_2;
        const T gp1_miu_3 = temp * gp1_term_3;

        value[0] += gp1_miu_1;
        value[1] += gp1_miu_2;
        value[2] += gp1_miu_3;

        // group 2
        const T gp2_term_1 = (181440.0 * P7x * P1y) - (952560.0 * P5x * P3y) - (952560.0 * P5x * P1y * P2z) + (793800.0 * P3x * P5y) + (1587600.0 * P3x * P3y * P2z) + (793800.0 * P3x * P1y * P4z) - (99225.0 * P1x * P7y) - (297675.0 * P1x * P5y * P2z) - (297675.0 * P1x * P3y * P4z) - (99225.0 * P1x * P1y * P6z);
        const T gp2_term_2 = (181440.0 * P7y * P1x) - (952560.0 * P5y * P3x) - (952560.0 * P5y * P1x * P2z) + (793800.0 * P3y * P5x) + (1587600.0 * P3y * P3x * P2z) + (793800.0 * P3y * P1x * P4z) - (99225.0 * P1y * P7x) - (297675.0 * P1y * P5x * P2z) - (297675.0 * P1y * P3x * P4z) - (99225.0 * P1y * P1x * P6z);
        const T gp2_term_3 = (181440.0 * P7x * P1z) - (952560.0 * P5x * P3z) - (952560.0 * P5x * P1z * P2y) + (793800.0 * P3x * P5z) + (1587600.0 * P3x * P3z * P2y) + (793800.0 * P3x * P1z * P4y) - (99225.0 * P1x * P7z) - (297675.0 * P1x * P5z * P2y) - (297675.0 * P1x * P3z * P4y) - (99225.0 * P1x * P1z * P6y);
        const T gp2_term_4 = (181440.0 * P7z * P1x) - (952560.0 * P5z * P3x) - (952560.0 * P5z * P1x * P2y) + (793800.0 * P3z * P5x) + (1587600.0 * P3z * P3x * P2y) + (793800.0 * P3z * P1x * P4y) - (99225.0 * P1z * P7x) - (297675.0 * P1z * P5x * P2y) - (297675.0 * P1z * P3x * P4y) - (99225.0 * P1z * P1x * P6y);
        const T gp2_term_5 = (181440.0 * P7y * P1z) - (952560.0 * P5y * P3z) - (952560.0 * P5y * P1z * P2x) + (793800.0 * P3y * P5z) + (1587600.0 * P3y * P3z * P2x) + (793800.0 * P3y * P1z * P4x) - (99225.0 * P1y * P7z) - (297675.0 * P1y * P5z * P2x) - (297675.0 * P1y * P3z * P4x) - (99225.0 * P1y * P1z * P6x);
        const T gp2_term_6 = (181440.0 * P7z * P1y) - (952560.0 * P5z * P3y) - (952560.0 * P5z * P1y * P2x) + (793800.0 * P3z * P5y) + (1587600.0 * P3z * P3y * P2x) + (793800.0 * P3z * P1y * P4x) - (99225.0 * P1z * P7y) - (297675.0 * P1z * P5y * P2x) - (297675.0 * P1z * P3y * P4x) - (99225.0 * P1z * P1y * P6x);

        const T gp2_miu_1 = temp * gp2_term_1;
        const T gp2_miu_2 = temp * gp2_term_2;
        const T gp2_miu_3 = temp * gp2_term_3;
        const T gp2_miu_4 = temp * gp2_term_4;
        const T gp2_miu_5 = temp * gp2_term_5;
        const T gp2_miu_6 = temp * gp2_term_6;

        value[3] += gp2_miu_1;
        value[4] += gp2_miu_2;
        value[5] += gp2_miu_3;
        value[6] += gp2_miu_4;
        value[7] += gp2_miu_5;
        value[8] += gp2_miu_6;


        // group 3
        const T gp3_term_1 = (509040.0 * P6x * P2y) - (20160.0 * P8x) + (55440.0 * P6x * P2z) - (1096200.0 * P4x * P4y) - (1058400.0 * P4x * P2y * P2z) + (37800.0 * P4x * P4z) + (389025.0 * P2x * P6y) + (741825.0 * P2x * P4y * P2z) + (316575.0 * P2x * P2y * P4z) - (36225.0 * P2x * P6z) - (12600.0 * P8y) - (36225.0 * P6y * P2z) - (33075.0 * P4y * P4z) - (7875.0 * P2y * P6z) + (1575.0 * P8z);
        const T gp3_term_2 = (509040.0 * P6y * P2x) - (20160.0 * P8y) + (55440.0 * P6y * P2z) - (1096200.0 * P4y * P4x) - (1058400.0 * P4y * P2x * P2z) + (37800.0 * P4y * P4z) + (389025.0 * P2y * P6x) + (741825.0 * P2y * P4x * P2z) + (316575.0 * P2y * P2x * P4z) - (36225.0 * P2y * P6z) - (12600.0 * P8x) - (36225.0 * P6x * P2z) - (33075.0 * P4x * P4z) - (7875.0 * P2x * P6z) + (1575.0 * P8z);
        const T gp3_term_3 = (509040.0 * P6x * P2z) - (20160.0 * P8x) + (55440.0 * P6x * P2y) - (1096200.0 * P4x * P4z) - (1058400.0 * P4x * P2z * P2y) + (37800.0 * P4x * P4y) + (389025.0 * P2x * P6z) + (741825.0 * P2x * P4z * P2y) + (316575.0 * P2x * P2z * P4y) - (36225.0 * P2x * P6y) - (12600.0 * P8z) - (36225.0 * P6z * P2y) - (33075.0 * P4z * P4y) - (7875.0 * P2z * P6y) + (1575.0 * P8y);
        const T gp3_term_4 = (509040.0 * P6z * P2x) - (20160.0 * P8z) + (55440.0 * P6z * P2y) - (1096200.0 * P4z * P4x) - (1058400.0 * P4z * P2x * P2y) + (37800.0 * P4z * P4y) + (389025.0 * P2z * P6x) + (741825.0 * P2z * P4x * P2y) + (316575.0 * P2z * P2x * P4y) - (36225.0 * P2z * P6y) - (12600.0 * P8x) - (36225.0 * P6x * P2y) - (33075.0 * P4x * P4y) - (7875.0 * P2x * P6y) + (1575.0 * P8y);
        const T gp3_term_5 = (509040.0 * P6y * P2z) - (20160.0 * P8y) + (55440.0 * P6y * P2x) - (1096200.0 * P4y * P4z) - (1058400.0 * P4y * P2z * P2x) + (37800.0 * P4y * P4x) + (389025.0 * P2y * P6z) + (741825.0 * P2y * P4z * P2x) + (316575.0 * P2y * P2z * P4x) - (36225.0 * P2y * P6x) - (12600.0 * P8z) - (36225.0 * P6z * P2x) - (33075.0 * P4z * P4x) - (7875.0 * P2z * P6x) + (1575.0 * P8x);
        const T gp3_term_6 = (509040.0 * P6z * P2y) - (20160.0 * P8z) + (55440.0 * P6z * P2x) - (1096200.0 * P4z * P4y) - (1058400.0 * P4z * P2y * P2x) + (37800.0 * P4z * P4x) + (389025.0 * P2z * P6y) + (741825.0 * P2z * P4y * P2x) + (316575.0 * P2z * P2y * P4x) - (36225.0 * P2z * P6x) - (12600.0 * P8y) - (36225.0 * P6y * P2x) - (33075.0 * P4y * P4x) - (7875.0 * P2y * P6x) + (1575.0 * P8x);

        const T gp3_miu_1 = temp * gp3_term_1;
        const T gp3_miu_2 = temp * gp3_term_2;
        const T gp3_miu_3 = temp * gp3_term_3;
        const T gp3_miu_4 = temp * gp3_term_4;
        const T gp3_miu_5 = temp * gp3_term_5;
        const T gp3_miu_6 = temp * gp3_term_6;

        value[9] += gp3_miu_1;
        value[10] += gp3_miu_2;
        value[11] += gp3_miu_3;
        value[12] += gp3_miu_4;
        value[13] += gp3_miu_5;
        value[14] += gp3_miu_6;


        // group 4
        const T gp4_term_1 = (453600.0 * P6x * P1y * P1z) - (1134000.0 * P4x * P3y * P1z) - (1134000.0 * P4x * P1y * P3z) + (425250.0 * P2x * P5y * P1z) + (850500.0 * P2x * P3y * P3z) + (425250.0 * P2x * P1y * P5z) - (14175.0 * P7y * P1z) - (42525.0 * P5y * P3z) - (42525.0 * P3y * P5z) - (14175.0 * P1y * P7z);
        const T gp4_term_2 = (453600.0 * P6y * P1x * P1z) - (1134000.0 * P4y * P3x * P1z) - (1134000.0 * P4y * P1x * P3z) + (425250.0 * P2y * P5x * P1z) + (850500.0 * P2y * P3x * P3z) + (425250.0 * P2y * P1x * P5z) - (14175.0 * P7x * P1z) - (42525.0 * P5x * P3z) - (42525.0 * P3x * P5z) - (14175.0 * P1x * P7z);
        const T gp4_term_3 = (453600.0 * P6z * P1x * P1y) - (1134000.0 * P4z * P3x * P1y) - (1134000.0 * P4z * P1x * P3y) + (425250.0 * P2z * P5x * P1y) + (850500.0 * P2z * P3x * P3y) + (425250.0 * P2z * P1x * P5y) - (14175.0 * P7x * P1y) - (42525.0 * P5x * P3y) - (42525.0 * P3x * P5y) - (14175.0 * P1x * P7y);

        const T gp4_miu_1 = temp * gp4_term_1;
        const T gp4_miu_2 = temp * gp4_term_2;
        const T gp4_miu_3 = temp * gp4_term_3;

        value[15] += gp4_miu_1;
        value[16] += gp4_miu_2;
        value[17] += gp4_miu_3;


        // group 5
        const T gp5_term_1 = (922320.0 * P5x * P3y) - (136080.0 * P7x * P1y) + (90720.0 * P5x * P1y * P2z) - (855225.0 * P3x * P5y) - (670950.0 * P3x * P3y * P2z) + (184275.0 * P3x * P1y * P4z) + (113400.0 * P1x * P7y) + (184275.0 * P1x * P5y * P2z) + (28350.0 * P1x * P3y * P4z) - (42525.0 * P1x * P1y * P6z);
        const T gp5_term_2 = (922320.0 * P5y * P3x) - (136080.0 * P7y * P1x) + (90720.0 * P5y * P1x * P2z) - (855225.0 * P3y * P5x) - (670950.0 * P3y * P3x * P2z) + (184275.0 * P3y * P1x * P4z) + (113400.0 * P1y * P7x) + (184275.0 * P1y * P5x * P2z) + (28350.0 * P1y * P3x * P4z) - (42525.0 * P1y * P1x * P6z);
        const T gp5_term_3 = (922320.0 * P5x * P3z) - (136080.0 * P7x * P1z) + (90720.0 * P5x * P1z * P2y) - (855225.0 * P3x * P5z) - (670950.0 * P3x * P3z * P2y) + (184275.0 * P3x * P1z * P4y) + (113400.0 * P1x * P7z) + (184275.0 * P1x * P5z * P2y) + (28350.0 * P1x * P3z * P4y) - (42525.0 * P1x * P1z * P6y);
        const T gp5_term_4 = (922320.0 * P5z * P3x) - (136080.0 * P7z * P1x) + (90720.0 * P5z * P1x * P2y) - (855225.0 * P3z * P5x) - (670950.0 * P3z * P3x * P2y) + (184275.0 * P3z * P1x * P4y) + (113400.0 * P1z * P7x) + (184275.0 * P1z * P5x * P2y) + (28350.0 * P1z * P3x * P4y) - (42525.0 * P1z * P1x * P6y);
        const T gp5_term_5 = (922320.0 * P5y * P3z) - (136080.0 * P7y * P1z) + (90720.0 * P5y * P1z * P2x) - (855225.0 * P3y * P5z) - (670950.0 * P3y * P3z * P2x) + (184275.0 * P3y * P1z * P4x) + (113400.0 * P1y * P7z) + (184275.0 * P1y * P5z * P2x) + (28350.0 * P1y * P3z * P4x) - (42525.0 * P1y * P1z * P6x);
        const T gp5_term_6 = (922320.0 * P5z * P3y) - (136080.0 * P7z * P1y) + (90720.0 * P5z * P1y * P2x) - (855225.0 * P3z * P5y) - (670950.0 * P3z * P3y * P2x) + (184275.0 * P3z * P1y * P4x) + (113400.0 * P1z * P7y) + (184275.0 * P1z * P5y * P2x) + (28350.0 * P1z * P3y * P4x) - (42525.0 * P1z * P1y * P6x);

        const T gp5_miu_1 = temp * gp5_term_1;
        const T gp5_miu_2 = temp * gp5_term_2;
        const T gp5_miu_3 = temp * gp5_term_3;
        const T gp5_miu_4 = temp * gp5_term_4;
        const T gp5_miu_5 = temp * gp5_term_5;
        const T gp5_miu_6 = temp * gp5_term_6;

        value[18] += gp5_miu_1;
        value[19] += gp5_miu_2;
        value[20] += gp5_miu_3;
        value[21] += gp5_miu_4;
        value[22] += gp5_miu_5;
        value[23] += gp5_miu_6;


        // group 6
        const T gp6_term_1 = (861840.0 * P5x * P2y * P1z) - (45360.0 * P7x * P1z) + (30240.0 * P5x * P3z) - (978075.0 * P3x * P4y * P1z) - (916650.0 * P3x * P2y * P3z) + (61425.0 * P3x * P5z) + (141750.0 * P1x * P6y * P1z) + (269325.0 * P1x * P4y * P3z) + (113400.0 * P1x * P2y * P5z) - (14175.0 * P1x * P7z);
        const T gp6_term_2 = (861840.0 * P5y * P2x * P1z) - (45360.0 * P7y * P1z) + (30240.0 * P5y * P3z) - (978075.0 * P3y * P4x * P1z) - (916650.0 * P3y * P2x * P3z) + (61425.0 * P3y * P5z) + (141750.0 * P1y * P6x * P1z) + (269325.0 * P1y * P4x * P3z) + (113400.0 * P1y * P2x * P5z) - (14175.0 * P1y * P7z);
        const T gp6_term_3 = (861840.0 * P5x * P2z * P1y) - (45360.0 * P7x * P1y) + (30240.0 * P5x * P3y) - (978075.0 * P3x * P4z * P1y) - (916650.0 * P3x * P2z * P3y) + (61425.0 * P3x * P5y) + (141750.0 * P1x * P6z * P1y) + (269325.0 * P1x * P4z * P3y) + (113400.0 * P1x * P2z * P5y) - (14175.0 * P1x * P7y);
        const T gp6_term_4 = (861840.0 * P5z * P2x * P1y) - (45360.0 * P7z * P1y) + (30240.0 * P5z * P3y) - (978075.0 * P3z * P4x * P1y) - (916650.0 * P3z * P2x * P3y) + (61425.0 * P3z * P5y) + (141750.0 * P1z * P6x * P1y) + (269325.0 * P1z * P4x * P3y) + (113400.0 * P1z * P2x * P5y) - (14175.0 * P1z * P7y);
        const T gp6_term_5 = (861840.0 * P5y * P2z * P1x) - (45360.0 * P7y * P1x) + (30240.0 * P5y * P3x) - (978075.0 * P3y * P4z * P1x) - (916650.0 * P3y * P2z * P3x) + (61425.0 * P3y * P5x) + (141750.0 * P1y * P6z * P1x) + (269325.0 * P1y * P4z * P3x) + (113400.0 * P1y * P2z * P5x) - (14175.0 * P1y * P7x);
        const T gp6_term_6 = (861840.0 * P5z * P2y * P1x) - (45360.0 * P7z * P1x) + (30240.0 * P5z * P3x) - (978075.0 * P3z * P4y * P1x) - (916650.0 * P3z * P2y * P3x) + (61425.0 * P3z * P5x) + (141750.0 * P1z * P6y * P1x) + (269325.0 * P1z * P4y * P3x) + (113400.0 * P1z * P2y * P5x) - (14175.0 * P1z * P7x);

        const T gp6_miu_1 = temp * gp6_term_1;
        const T gp6_miu_2 = temp * gp6_term_2;
        const T gp6_miu_3 = temp * gp6_term_3;
        const T gp6_miu_4 = temp * gp6_term_4;
        const T gp6_miu_5 = temp * gp6_term_5;
        const T gp6_miu_6 = temp * gp6_term_6;

        value[24] += gp6_miu_1;
        value[25] += gp6_miu_2;
        value[26] += gp6_miu_3;
        value[27] += gp6_miu_4;
        value[28] += gp6_miu_5;
        value[29] += gp6_miu_6;


        // group 7
        const T gp7_term_1 = (1119825.0 * P4x * P4y) - (438480.0 * P6x * P2y) - (141750.0 * P4x * P2y * P2z) + (15120.0 * P8x) + (15120.0 * P6x * P2z) - (14175.0 * P4x * P4z) - (438480.0 * P2x * P6y) - (141750.0 * P2x * P4y * P2z) + (283500.0 * P2x * P2y * P4z) - (13230.0 * P2x * P6z) + (15120.0 * P8y) + (15120.0 * P6y * P2z) - (14175.0 * P4y * P4z) - (13230.0 * P2y * P6z) + (945.0 * P8z);
        const T gp7_term_2 = (1119825.0 * P4x * P4z) - (438480.0 * P6x * P2z) - (141750.0 * P4x * P2z * P2y) + (15120.0 * P8x) + (15120.0 * P6x * P2y) - (14175.0 * P4x * P4y) - (438480.0 * P2x * P6z) - (141750.0 * P2x * P4z * P2y) + (283500.0 * P2x * P2z * P4y) - (13230.0 * P2x * P6y) + (15120.0 * P8z) + (15120.0 * P6z * P2y) - (14175.0 * P4z * P4y) - (13230.0 * P2z * P6y) + (945.0 * P8y);
        const T gp7_term_3 = (1119825.0 * P4y * P4z) - (438480.0 * P6y * P2z) - (141750.0 * P4y * P2z * P2x) + (15120.0 * P8y) + (15120.0 * P6y * P2x) - (14175.0 * P4y * P4x) - (438480.0 * P2y * P6z) - (141750.0 * P2y * P4z * P2x) + (283500.0 * P2y * P2z * P4x) - (13230.0 * P2y * P6x) + (15120.0 * P8z) + (15120.0 * P6z * P2x) - (14175.0 * P4z * P4x) - (13230.0 * P2z * P6x) + (945.0 * P8x);

        const T gp7_miu_1 = temp * gp7_term_1;
        const T gp7_miu_2 = temp * gp7_term_2;
        const T gp7_miu_3 = temp * gp7_term_3;

        value[30] += gp7_miu_1;
        value[31] += gp7_miu_2;
        value[32] += gp7_miu_3;


        // group 8
        const T gp8_term_1 = (1190700.0 * P4x * P3y * P1z) - (226800.0 * P6x * P1y * P1z) - (56700.0 * P4x * P1y * P3z) - (586845.0 * P2x * P5y * P1z) - (425250.0 * P2x * P3y * P3z) + (161595.0 * P2x * P1y * P5z) + (22680.0 * P7y * P1z) + (36855.0 * P5y * P3z) + (5670.0 * P3y * P5z) - (8505.0 * P1y * P7z);
        const T gp8_term_2 = (1190700.0 * P4y * P3x * P1z) - (226800.0 * P6y * P1x * P1z) - (56700.0 * P4y * P1x * P3z) - (586845.0 * P2y * P5x * P1z) - (425250.0 * P2y * P3x * P3z) + (161595.0 * P2y * P1x * P5z) + (22680.0 * P7x * P1z) + (36855.0 * P5x * P3z) + (5670.0 * P3x * P5z) - (8505.0 * P1x * P7z);
        const T gp8_term_3 = (1190700.0 * P4x * P3z * P1y) - (226800.0 * P6x * P1z * P1y) - (56700.0 * P4x * P1z * P3y) - (586845.0 * P2x * P5z * P1y) - (425250.0 * P2x * P3z * P3y) + (161595.0 * P2x * P1z * P5y) + (22680.0 * P7z * P1y) + (36855.0 * P5z * P3y) + (5670.0 * P3z * P5y) - (8505.0 * P1z * P7y);
        const T gp8_term_4 = (1190700.0 * P4z * P3x * P1y) - (226800.0 * P6z * P1x * P1y) - (56700.0 * P4z * P1x * P3y) - (586845.0 * P2z * P5x * P1y) - (425250.0 * P2z * P3x * P3y) + (161595.0 * P2z * P1x * P5y) + (22680.0 * P7x * P1y) + (36855.0 * P5x * P3y) + (5670.0 * P3x * P5y) - (8505.0 * P1x * P7y);
        const T gp8_term_5 = (1190700.0 * P4y * P3z * P1x) - (226800.0 * P6y * P1z * P1x) - (56700.0 * P4y * P1z * P3x) - (586845.0 * P2y * P5z * P1x) - (425250.0 * P2y * P3z * P3x) + (161595.0 * P2y * P1z * P5x) + (22680.0 * P7z * P1x) + (36855.0 * P5z * P3x) + (5670.0 * P3z * P5x) - (8505.0 * P1z * P7x);
        const T gp8_term_6 = (1190700.0 * P4z * P3y * P1x) - (226800.0 * P6z * P1y * P1x) - (56700.0 * P4z * P1y * P3x) - (586845.0 * P2z * P5y * P1x) - (425250.0 * P2z * P3y * P3x) + (161595.0 * P2z * P1y * P5x) + (22680.0 * P7y * P1x) + (36855.0 * P5y * P3x) + (5670.0 * P3y * P5x) - (8505.0 * P1y * P7x);

        const T gp8_miu_1 = temp * gp8_term_1;
        const T gp8_miu_2 = temp * gp8_term_2;
        const T gp8_miu_3 = temp * gp8_term_3;
        const T gp8_miu_4 = temp * gp8_term_4;
        const T gp8_miu_5 = temp * gp8_term_5;
        const T gp8_miu_6 = temp * gp8_term_6;

        value[33] += gp8_miu_1;
        value[34] += gp8_miu_2;
        value[35] += gp8_miu_3;
        value[36] += gp8_miu_4;
        value[37] += gp8_miu_5;
        value[38] += gp8_miu_6;


        // group 9
        const T gp9_term_1 = (1200150.0 * P4x * P2y * P2z) - (70560.0 * P6x * P2y) - (23625.0 * P4x * P4y) - (70560.0 * P6x * P2z) - (23625.0 * P4x * P4z) + (5040.0 * P8x) - (600075.0 * P2x * P4y * P2z) - (600075.0 * P2x * P2y * P4z) + (49455.0 * P2x * P6y) + (49455.0 * P2x * P6z) + (21105.0 * P6y * P2z) + (47250.0 * P4y * P4z) + (21105.0 * P2y * P6z) - (2520.0 * P8y) - (2520.0 * P8z);
        const T gp9_term_2 = (1200150.0 * P4y * P2x * P2z) - (70560.0 * P6y * P2x) - (23625.0 * P4y * P4x) - (70560.0 * P6y * P2z) - (23625.0 * P4y * P4z) + (5040.0 * P8y) - (600075.0 * P2y * P4x * P2z) - (600075.0 * P2y * P2x * P4z) + (49455.0 * P2y * P6x) + (49455.0 * P2y * P6z) + (21105.0 * P6x * P2z) + (47250.0 * P4x * P4z) + (21105.0 * P2x * P6z) - (2520.0 * P8x) - (2520.0 * P8z);
        const T gp9_term_3 = (1200150.0 * P4z * P2x * P2y) - (70560.0 * P6z * P2x) - (23625.0 * P4z * P4x) - (70560.0 * P6z * P2y) - (23625.0 * P4z * P4y) + (5040.0 * P8z) - (600075.0 * P2z * P4x * P2y) - (600075.0 * P2z * P2x * P4y) + (49455.0 * P2z * P6x) + (49455.0 * P2z * P6y) + (21105.0 * P6x * P2y) + (47250.0 * P4x * P4y) + (21105.0 * P2x * P6y) - (2520.0 * P8x) - (2520.0 * P8y);

        const T gp9_miu_1 = temp * gp9_term_1;
        const T gp9_miu_2 = temp * gp9_term_2;
        const T gp9_miu_3 = temp * gp9_term_3;

        value[39] += gp9_miu_1;
        value[40] += gp9_miu_2;
        value[41] += gp9_miu_3;


        // group 10
        const T gp10_term_1 = (1341900.0 * P3x * P3y * P2z) - (67095.0 * P5x * P3y) - (67095.0 * P3x * P5y) - (274995.0 * P5x * P1y * P2z) - (212625.0 * P3x * P1y * P4z) + (22680.0 * P7x * P1y) - (274995.0 * P1x * P5y * P2z) - (212625.0 * P1x * P3y * P4z) + (22680.0 * P1x * P7y) + (85050.0 * P1x * P1y * P6z);
        const T gp10_term_2 = (1341900.0 * P3x * P3z * P2y) - (67095.0 * P5x * P3z) - (67095.0 * P3x * P5z) - (274995.0 * P5x * P1z * P2y) - (212625.0 * P3x * P1z * P4y) + (22680.0 * P7x * P1z) - (274995.0 * P1x * P5z * P2y) - (212625.0 * P1x * P3z * P4y) + (22680.0 * P1x * P7z) + (85050.0 * P1x * P1z * P6y);
        const T gp10_term_3 = (1341900.0 * P3y * P3z * P2x) - (67095.0 * P5y * P3z) - (67095.0 * P3y * P5z) - (274995.0 * P5y * P1z * P2x) - (212625.0 * P3y * P1z * P4x) + (22680.0 * P7y * P1z) - (274995.0 * P1y * P5z * P2x) - (212625.0 * P1y * P3z * P4x) + (22680.0 * P1y * P7z) + (85050.0 * P1y * P1z * P6x);

        const T gp10_miu_1 = temp * gp10_term_1;
        const T gp10_miu_2 = temp * gp10_term_2;
        const T gp10_miu_3 = temp * gp10_term_3;

        value[42] += gp10_miu_1;
        value[43] += gp10_miu_2;
        value[44] += gp10_miu_3;

    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_9(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const T lambda_x0 = lambda * dr[0];
        const T lambda_x0_2 = lambda_x0 * lambda_x0;
        const T lambda_x0_3 = lambda_x0_2 * lambda_x0;
        const T lambda_x0_4 = lambda_x0_3 * lambda_x0;
        const T lambda_x0_5 = lambda_x0_4 * lambda_x0;
        const T lambda_x0_6 = lambda_x0_5 * lambda_x0;
        const T lambda_x0_7 = lambda_x0_6 * lambda_x0;
        const T lambda_x0_8 = lambda_x0_7 * lambda_x0;
        const T lambda_x0_9 = lambda_x0_8 * lambda_x0;

        const T lambda_y0 = lambda * dr[1];
        const T lambda_y0_2 = lambda_y0 * lambda_y0;
        const T lambda_y0_3 = lambda_y0_2 * lambda_y0;
        const T lambda_y0_4 = lambda_y0_3 * lambda_y0;
        const T lambda_y0_5 = lambda_y0_4 * lambda_y0;
        const T lambda_y0_6 = lambda_y0_5 * lambda_y0;
        const T lambda_y0_7 = lambda_y0_6 * lambda_y0;
        const T lambda_y0_8 = lambda_y0_7 * lambda_y0;
        const T lambda_y0_9 = lambda_y0_8 * lambda_y0;

        const T lambda_z0 = lambda * dr[2];
        const T lambda_z0_2 = lambda_z0 * lambda_z0;
        const T lambda_z0_3 = lambda_z0_2 * lambda_z0;
        const T lambda_z0_4 = lambda_z0_3 * lambda_z0;
        const T lambda_z0_5 = lambda_z0_4 * lambda_z0;
        const T lambda_z0_6 = lambda_z0_5 * lambda_z0;
        const T lambda_z0_7 = lambda_z0_6 * lambda_z0;
        const T lambda_z0_8 = lambda_z0_7 * lambda_z0;
        const T lambda_z0_9 = lambda_z0_8 * lambda_z0;

        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;
        const T inv_gamma_3 = inv_gamma_2 * inv_gamma;
        const T inv_gamma_4 = inv_gamma_3 * inv_gamma;

        const T P1x = lambda_x0;
        const T P1y = lambda_y0;
        const T P1z = lambda_z0;

        const T P2x = P2(lambda_x0_2, inv_gamma);
        const T P2y = P2(lambda_y0_2, inv_gamma);
        const T P2z = P2(lambda_z0_2, inv_gamma);

        const T P3x = P3(lambda_x0, lambda_x0_3, inv_gamma);
        const T P3y = P3(lambda_y0, lambda_y0_3, inv_gamma);
        const T P3z = P3(lambda_z0, lambda_z0_3, inv_gamma);

        const T P4x = P4(lambda_x0_2, lambda_x0_4, inv_gamma, inv_gamma_2);
        const T P4y = P4(lambda_y0_2, lambda_y0_4, inv_gamma, inv_gamma_2);
        const T P4z = P4(lambda_z0_2, lambda_z0_4, inv_gamma, inv_gamma_2);

        const T P5x = P5(lambda_x0, lambda_x0_3, lambda_x0_5, inv_gamma, inv_gamma_2);
        const T P5y = P5(lambda_y0, lambda_y0_3, lambda_y0_5, inv_gamma, inv_gamma_2);
        const T P5z = P5(lambda_z0, lambda_z0_3, lambda_z0_5, inv_gamma, inv_gamma_2);

        const T P6x = P6(lambda_x0_2, lambda_x0_4, lambda_x0_6, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P6y = P6(lambda_y0_2, lambda_y0_4, lambda_y0_6, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P6z = P6(lambda_z0_2, lambda_z0_4, lambda_z0_6, inv_gamma, inv_gamma_2, inv_gamma_3);

        const T P7x = P7(lambda_x0, lambda_x0_3, lambda_x0_5, lambda_x0_7, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P7y = P7(lambda_y0, lambda_y0_3, lambda_y0_5, lambda_y0_7, inv_gamma, inv_gamma_2, inv_gamma_3);
        const T P7z = P7(lambda_z0, lambda_z0_3, lambda_z0_5, lambda_z0_7, inv_gamma, inv_gamma_2, inv_gamma_3);

        const T P8x = P8(lambda_x0_2, lambda_x0_4, lambda_x0_6, lambda_x0_8, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
        const T P8y = P8(lambda_y0_2, lambda_y0_4, lambda_y0_6, lambda_y0_8, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
        const T P8z = P8(lambda_z0_2, lambda_z0_4, lambda_z0_6, lambda_z0_8, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);

        const T P9x = P9(lambda_x0, lambda_x0_3, lambda_x0_5, lambda_x0_7, lambda_x0_9, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
        const T P9y = P9(lambda_y0, lambda_y0_3, lambda_y0_5, lambda_y0_7, lambda_y0_9, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
        const T P9z = P9(lambda_z0, lambda_z0_3, lambda_z0_5, lambda_z0_7, lambda_z0_9, inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);

        // group 1
        const T gp1_term_1 = (362880.0 * P9x) - (6531840.0 * P7x * P2y) - (6531840.0 * P7x * P2z) + (17146080.0 * P5x * P4y) + (34292160.0 * P5x * P2y * P2z) + (17146080.0 * P5x * P4z) - (9525600.0 * P3x * P6y) - (28576800.0 * P3x * P4y * P2z) - (28576800.0 * P3x * P2y * P4z) - (9525600.0 * P3x * P6z) + (893025.0 * P1x * P8y) + (3572100.0 * P1x * P6y * P2z) + (5358150.0 * P1x * P4y * P4z) + (3572100.0 * P1x * P2y * P6z) + (893025.0 * P1x * P8z);
        const T gp1_term_2 = (362880.0 * P9y) - (6531840.0 * P7y * P2x) - (6531840.0 * P7y * P2z) + (17146080.0 * P5y * P4x) + (34292160.0 * P5y * P2x * P2z) + (17146080.0 * P5y * P4z) - (9525600.0 * P3y * P6x) - (28576800.0 * P3y * P4x * P2z) - (28576800.0 * P3y * P2x * P4z) - (9525600.0 * P3y * P6z) + (893025.0 * P1y * P8x) + (3572100.0 * P1y * P6x * P2z) + (5358150.0 * P1y * P4x * P4z) + (3572100.0 * P1y * P2x * P6z) + (893025.0 * P1y * P8z);
        const T gp1_term_3 = (362880.0 * P9z) - (6531840.0 * P7z * P2x) - (6531840.0 * P7z * P2y) + (17146080.0 * P5z * P4x) + (34292160.0 * P5z * P2x * P2y) + (17146080.0 * P5z * P4y) - (9525600.0 * P3z * P6x) - (28576800.0 * P3z * P4x * P2y) - (28576800.0 * P3z * P2x * P4y) - (9525600.0 * P3z * P6y) + (893025.0 * P1z * P8x) + (3572100.0 * P1z * P6x * P2y) + (5358150.0 * P1z * P4x * P4y) + (3572100.0 * P1z * P2x * P6y) + (893025.0 * P1z * P8y);

        const T gp1_miu_1 = temp * gp1_term_1;
        const T gp1_miu_2 = temp * gp1_term_2;
        const T gp1_miu_3 = temp * gp1_term_3;

        value[0] += gp1_miu_1;
        value[1] += gp1_miu_2;
        value[2] += gp1_miu_3;

        // group 2
        const T gp2_term_1 = (1814400.0 * P8x * P1y) - (12700800.0 * P6x * P3y) - (12700800.0 * P6x * P1y * P2z) + (15876000.0 * P4x * P5y) + (31752000.0 * P4x * P3y * P2z) + (15876000.0 * P4x * P1y * P4z) - (3969000.0 * P2x * P7y) - (11907000.0 * P2x * P5y * P2z) - (11907000.0 * P2x * P3y * P4z) - (3969000.0 * P2x * P1y * P6z) + (99225.0 * P9y) + (396900.0 * P7y * P2z) + (595350.0 * P5y * P4z) + (396900.0 * P3y * P6z) + (99225.0 * P1y * P8z);
        const T gp2_term_2 = (1814400.0 * P8y * P1x) - (12700800.0 * P6y * P3x) - (12700800.0 * P6y * P1x * P2z) + (15876000.0 * P4y * P5x) + (31752000.0 * P4y * P3x * P2z) + (15876000.0 * P4y * P1x * P4z) - (3969000.0 * P2y * P7x) - (11907000.0 * P2y * P5x * P2z) - (11907000.0 * P2y * P3x * P4z) - (3969000.0 * P2y * P1x * P6z) + (99225.0 * P9x) + (396900.0 * P7x * P2z) + (595350.0 * P5x * P4z) + (396900.0 * P3x * P6z) + (99225.0 * P1x * P8z);
        const T gp2_term_3 = (1814400.0 * P8x * P1z) - (12700800.0 * P6x * P3z) - (12700800.0 * P6x * P1z * P2y) + (15876000.0 * P4x * P5z) + (31752000.0 * P4x * P3z * P2y) + (15876000.0 * P4x * P1z * P4y) - (3969000.0 * P2x * P7z) - (11907000.0 * P2x * P5z * P2y) - (11907000.0 * P2x * P3z * P4y) - (3969000.0 * P2x * P1z * P6y) + (99225.0 * P9z) + (396900.0 * P7z * P2y) + (595350.0 * P5z * P4y) + (396900.0 * P3z * P6y) + (99225.0 * P1z * P8y);
        const T gp2_term_4 = (1814400.0 * P8z * P1x) - (12700800.0 * P6z * P3x) - (12700800.0 * P6z * P1x * P2y) + (15876000.0 * P4z * P5x) + (31752000.0 * P4z * P3x * P2y) + (15876000.0 * P4z * P1x * P4y) - (3969000.0 * P2z * P7x) - (11907000.0 * P2z * P5x * P2y) - (11907000.0 * P2z * P3x * P4y) - (3969000.0 * P2z * P1x * P6y) + (99225.0 * P9x) + (396900.0 * P7x * P2y) + (595350.0 * P5x * P4y) + (396900.0 * P3x * P6y) + (99225.0 * P1x * P8y);
        const T gp2_term_5 = (1814400.0 * P8y * P1z) - (12700800.0 * P6y * P3z) - (12700800.0 * P6y * P1z * P2x) + (15876000.0 * P4y * P5z) + (31752000.0 * P4y * P3z * P2x) + (15876000.0 * P4y * P1z * P4x) - (3969000.0 * P2y * P7z) - (11907000.0 * P2y * P5z * P2x) - (11907000.0 * P2y * P3z * P4x) - (3969000.0 * P2y * P1z * P6x) + (99225.0 * P9z) + (396900.0 * P7z * P2x) + (595350.0 * P5z * P4x) + (396900.0 * P3z * P6x) + (99225.0 * P1z * P8x);
        const T gp2_term_6 = (1814400.0 * P8z * P1y) - (12700800.0 * P6z * P3y) - (12700800.0 * P6z * P1y * P2x) + (15876000.0 * P4z * P5y) + (31752000.0 * P4z * P3y * P2x) + (15876000.0 * P4z * P1y * P4x) - (3969000.0 * P2z * P7y) - (11907000.0 * P2z * P5y * P2x) - (11907000.0 * P2z * P3y * P4x) - (3969000.0 * P2z * P1y * P6x) + (99225.0 * P9y) + (396900.0 * P7y * P2x) + (595350.0 * P5y * P4x) + (396900.0 * P3y * P6x) + (99225.0 * P1y * P8x);

        const T gp2_miu_1 = temp * gp2_term_1;
        const T gp2_miu_2 = temp * gp2_term_2;
        const T gp2_miu_3 = temp * gp2_term_3;
        const T gp2_miu_4 = temp * gp2_term_4;
        const T gp2_miu_5 = temp * gp2_term_5;
        const T gp2_miu_6 = temp * gp2_term_6;

        value[3] += gp2_miu_1;
        value[4] += gp2_miu_2;
        value[5] += gp2_miu_3;
        value[6] += gp2_miu_4;
        value[7] += gp2_miu_5;
        value[8] += gp2_miu_6;


        // group 3
        const T gp3_term_1 = (5760720.0 * P7x * P2y) - (181440.0 * P9x) + (771120.0 * P7x * P2z) - (17304840.0 * P5x * P4y) - (17146080.0 * P5x * P2y * P2z) + (158760.0 * P5x * P4z) + (10220175.0 * P3x * P6y) + (19745775.0 * P3x * P4y * P2z) + (8831025.0 * P3x * P2y * P4z) - (694575.0 * P3x * P6z) - (992250.0 * P1x * P8y) - (2877525.0 * P1x * P6y * P2z) - (2679075.0 * P1x * P4y * P4z) - (694575.0 * P1x * P2y * P6z) + (99225.0 * P1x * P8z);
        const T gp3_term_2 = (5760720.0 * P7y * P2x) - (181440.0 * P9y) + (771120.0 * P7y * P2z) - (17304840.0 * P5y * P4x) - (17146080.0 * P5y * P2x * P2z) + (158760.0 * P5y * P4z) + (10220175.0 * P3y * P6x) + (19745775.0 * P3y * P4x * P2z) + (8831025.0 * P3y * P2x * P4z) - (694575.0 * P3y * P6z) - (992250.0 * P1y * P8x) - (2877525.0 * P1y * P6x * P2z) - (2679075.0 * P1y * P4x * P4z) - (694575.0 * P1y * P2x * P6z) + (99225.0 * P1y * P8z);
        const T gp3_term_3 = (5760720.0 * P7x * P2z) - (181440.0 * P9x) + (771120.0 * P7x * P2y) - (17304840.0 * P5x * P4z) - (17146080.0 * P5x * P2z * P2y) + (158760.0 * P5x * P4y) + (10220175.0 * P3x * P6z) + (19745775.0 * P3x * P4z * P2y) + (8831025.0 * P3x * P2z * P4y) - (694575.0 * P3x * P6y) - (992250.0 * P1x * P8z) - (2877525.0 * P1x * P6z * P2y) - (2679075.0 * P1x * P4z * P4y) - (694575.0 * P1x * P2z * P6y) + (99225.0 * P1x * P8y);
        const T gp3_term_4 = (5760720.0 * P7z * P2x) - (181440.0 * P9z) + (771120.0 * P7z * P2y) - (17304840.0 * P5z * P4x) - (17146080.0 * P5z * P2x * P2y) + (158760.0 * P5z * P4y) + (10220175.0 * P3z * P6x) + (19745775.0 * P3z * P4x * P2y) + (8831025.0 * P3z * P2x * P4y) - (694575.0 * P3z * P6y) - (992250.0 * P1z * P8x) - (2877525.0 * P1z * P6x * P2y) - (2679075.0 * P1z * P4x * P4y) - (694575.0 * P1z * P2x * P6y) + (99225.0 * P1z * P8y);
        const T gp3_term_5 = (5760720.0 * P7y * P2z) - (181440.0 * P9y) + (771120.0 * P7y * P2x) - (17304840.0 * P5y * P4z) - (17146080.0 * P5y * P2z * P2x) + (158760.0 * P5y * P4x) + (10220175.0 * P3y * P6z) + (19745775.0 * P3y * P4z * P2x) + (8831025.0 * P3y * P2z * P4x) - (694575.0 * P3y * P6x) - (992250.0 * P1y * P8z) - (2877525.0 * P1y * P6z * P2x) - (2679075.0 * P1y * P4z * P4x) - (694575.0 * P1y * P2z * P6x) + (99225.0 * P1y * P8x);
        const T gp3_term_6 = (5760720.0 * P7z * P2y) - (181440.0 * P9z) + (771120.0 * P7z * P2x) - (17304840.0 * P5z * P4y) - (17146080.0 * P5z * P2y * P2x) + (158760.0 * P5z * P4x) + (10220175.0 * P3z * P6y) + (19745775.0 * P3z * P4y * P2x) + (8831025.0 * P3z * P2y * P4x) - (694575.0 * P3z * P6x) - (992250.0 * P1z * P8y) - (2877525.0 * P1z * P6y * P2x) - (2679075.0 * P1z * P4y * P4x) - (694575.0 * P1z * P2y * P6x) + (99225.0 * P1z * P8x);

        const T gp3_miu_1 = temp * gp3_term_1;
        const T gp3_miu_2 = temp * gp3_term_2;
        const T gp3_miu_3 = temp * gp3_term_3;
        const T gp3_miu_4 = temp * gp3_term_4;
        const T gp3_miu_5 = temp * gp3_term_5;
        const T gp3_miu_6 = temp * gp3_term_6;

        value[9] += gp3_miu_1;
        value[10] += gp3_miu_2;
        value[11] += gp3_miu_3;
        value[12] += gp3_miu_4;
        value[13] += gp3_miu_5;
        value[14] += gp3_miu_6;


        // group 4
        const T gp4_term_1 = (4989600.0 * P7x * P1y * P1z) - (17463600.0 * P5x * P3y * P1z) - (17463600.0 * P5x * P1y * P3z) + (10914750.0 * P3x * P5y * P1z) + (21829500.0 * P3x * P3y * P3z) + (10914750.0 * P3x * P1y * P5z) - (1091475.0 * P1x * P7y * P1z) - (3274425.0 * P1x * P5y * P3z) - (3274425.0 * P1x * P3y * P5z) - (1091475.0 * P1x * P1y * P7z);
        const T gp4_term_2 = (4989600.0 * P7y * P1x * P1z) - (17463600.0 * P5y * P3x * P1z) - (17463600.0 * P5y * P1x * P3z) + (10914750.0 * P3y * P5x * P1z) + (21829500.0 * P3y * P3x * P3z) + (10914750.0 * P3y * P1x * P5z) - (1091475.0 * P1y * P7x * P1z) - (3274425.0 * P1y * P5x * P3z) - (3274425.0 * P1y * P3x * P5z) - (1091475.0 * P1y * P1x * P7z);
        const T gp4_term_3 = (4989600.0 * P7z * P1x * P1y) - (17463600.0 * P5z * P3x * P1y) - (17463600.0 * P5z * P1x * P3y) + (10914750.0 * P3z * P5x * P1y) + (21829500.0 * P3z * P3x * P3y) + (10914750.0 * P3z * P1x * P5y) - (1091475.0 * P1z * P7x * P1y) - (3274425.0 * P1z * P5x * P3y) - (3274425.0 * P1z * P3x * P5y) - (1091475.0 * P1z * P1x * P7y);

        const T gp4_miu_1 = temp * gp4_term_1;
        const T gp4_miu_2 = temp * gp4_term_2;
        const T gp4_miu_3 = temp * gp4_term_3;

        value[15] += gp4_miu_1;
        value[16] += gp4_miu_2;
        value[17] += gp4_miu_3;


        // group 5
        const T gp5_term_1 = (12020400.0 * P6x * P3y) - (1360800.0 * P8x * P1y) + (2041200.0 * P6x * P1y * P2z) - (16584750.0 * P4x * P5y) - (14458500.0 * P4x * P3y * P2z) + (2126250.0 * P4x * P1y * P4z) + (4380075.0 * P2x * P7y) + (7526925.0 * P2x * P5y * P2z) + (1913625.0 * P2x * P3y * P4z) - (1233225.0 * P2x * P1y * P6z) - (113400.0 * P9y) - (297675.0 * P7y * P2z) - (212625.0 * P5y * P4z) + (14175.0 * P3y * P6z) + (42525.0 * P1y * P8z);
        const T gp5_term_2 = (12020400.0 * P6y * P3x) - (1360800.0 * P8y * P1x) + (2041200.0 * P6y * P1x * P2z) - (16584750.0 * P4y * P5x) - (14458500.0 * P4y * P3x * P2z) + (2126250.0 * P4y * P1x * P4z) + (4380075.0 * P2y * P7x) + (7526925.0 * P2y * P5x * P2z) + (1913625.0 * P2y * P3x * P4z) - (1233225.0 * P2y * P1x * P6z) - (113400.0 * P9x) - (297675.0 * P7x * P2z) - (212625.0 * P5x * P4z) + (14175.0 * P3x * P6z) + (42525.0 * P1x * P8z);
        const T gp5_term_3 = (12020400.0 * P6x * P3z) - (1360800.0 * P8x * P1z) + (2041200.0 * P6x * P1z * P2y) - (16584750.0 * P4x * P5z) - (14458500.0 * P4x * P3z * P2y) + (2126250.0 * P4x * P1z * P4y) + (4380075.0 * P2x * P7z) + (7526925.0 * P2x * P5z * P2y) + (1913625.0 * P2x * P3z * P4y) - (1233225.0 * P2x * P1z * P6y) - (113400.0 * P9z) - (297675.0 * P7z * P2y) - (212625.0 * P5z * P4y) + (14175.0 * P3z * P6y) + (42525.0 * P1z * P8y);
        const T gp5_term_4 = (12020400.0 * P6z * P3x) - (1360800.0 * P8z * P1x) + (2041200.0 * P6z * P1x * P2y) - (16584750.0 * P4z * P5x) - (14458500.0 * P4z * P3x * P2y) + (2126250.0 * P4z * P1x * P4y) + (4380075.0 * P2z * P7x) + (7526925.0 * P2z * P5x * P2y) + (1913625.0 * P2z * P3x * P4y) - (1233225.0 * P2z * P1x * P6y) - (113400.0 * P9x) - (297675.0 * P7x * P2y) - (212625.0 * P5x * P4y) + (14175.0 * P3x * P6y) + (42525.0 * P1x * P8y);
        const T gp5_term_5 = (12020400.0 * P6y * P3z) - (1360800.0 * P8y * P1z) + (2041200.0 * P6y * P1z * P2x) - (16584750.0 * P4y * P5z) - (14458500.0 * P4y * P3z * P2x) + (2126250.0 * P4y * P1z * P4x) + (4380075.0 * P2y * P7z) + (7526925.0 * P2y * P5z * P2x) + (1913625.0 * P2y * P3z * P4x) - (1233225.0 * P2y * P1z * P6x) - (113400.0 * P9z) - (297675.0 * P7z * P2x) - (212625.0 * P5z * P4x) + (14175.0 * P3z * P6x) + (42525.0 * P1z * P8x);
        const T gp5_term_6 = (12020400.0 * P6z * P3y) - (1360800.0 * P8z * P1y) + (2041200.0 * P6z * P1y * P2x) - (16584750.0 * P4z * P5y) - (14458500.0 * P4z * P3y * P2x) + (2126250.0 * P4z * P1y * P4x) + (4380075.0 * P2z * P7y) + (7526925.0 * P2z * P5y * P2x) + (1913625.0 * P2z * P3y * P4x) - (1233225.0 * P2z * P1y * P6x) - (113400.0 * P9y) - (297675.0 * P7y * P2x) - (212625.0 * P5y * P4x) + (14175.0 * P3y * P6x) + (42525.0 * P1y * P8x);

        const T gp5_miu_1 = temp * gp5_term_1;
        const T gp5_miu_2 = temp * gp5_term_2;
        const T gp5_miu_3 = temp * gp5_term_3;
        const T gp5_miu_4 = temp * gp5_term_4;
        const T gp5_miu_5 = temp * gp5_term_5;
        const T gp5_miu_6 = temp * gp5_term_6;

        value[18] += gp5_miu_1;
        value[19] += gp5_miu_2;
        value[20] += gp5_miu_3;
        value[21] += gp5_miu_4;
        value[22] += gp5_miu_5;
        value[23] += gp5_miu_6;


        // group 6
        const T gp6_term_1 = (10659600.0 * P6x * P2y * P1z) - (453600.0 * P8x * P1z) + (680400.0 * P6x * P3z) - (18002250.0 * P4x * P4y * P1z) - (17293500.0 * P4x * P2y * P3z) + (708750.0 * P4x * P5z) + (5202225.0 * P2x * P6y * P1z) + (9993375.0 * P2x * P4y * P3z) + (4380075.0 * P2x * P2y * P5z) - (411075.0 * P2x * P7z) - (141750.0 * P8y * P1z) - (411075.0 * P6y * P3z) - (382725.0 * P4y * P5z) - (99225.0 * P2y * P7z) + (14175.0 * P9z);
        const T gp6_term_2 = (10659600.0 * P6y * P2x * P1z) - (453600.0 * P8y * P1z) + (680400.0 * P6y * P3z) - (18002250.0 * P4y * P4x * P1z) - (17293500.0 * P4y * P2x * P3z) + (708750.0 * P4y * P5z) + (5202225.0 * P2y * P6x * P1z) + (9993375.0 * P2y * P4x * P3z) + (4380075.0 * P2y * P2x * P5z) - (411075.0 * P2y * P7z) - (141750.0 * P8x * P1z) - (411075.0 * P6x * P3z) - (382725.0 * P4x * P5z) - (99225.0 * P2x * P7z) + (14175.0 * P9z);
        const T gp6_term_3 = (10659600.0 * P6x * P2z * P1y) - (453600.0 * P8x * P1y) + (680400.0 * P6x * P3y) - (18002250.0 * P4x * P4z * P1y) - (17293500.0 * P4x * P2z * P3y) + (708750.0 * P4x * P5y) + (5202225.0 * P2x * P6z * P1y) + (9993375.0 * P2x * P4z * P3y) + (4380075.0 * P2x * P2z * P5y) - (411075.0 * P2x * P7y) - (141750.0 * P8z * P1y) - (411075.0 * P6z * P3y) - (382725.0 * P4z * P5y) - (99225.0 * P2z * P7y) + (14175.0 * P9y);
        const T gp6_term_4 = (10659600.0 * P6z * P2x * P1y) - (453600.0 * P8z * P1y) + (680400.0 * P6z * P3y) - (18002250.0 * P4z * P4x * P1y) - (17293500.0 * P4z * P2x * P3y) + (708750.0 * P4z * P5y) + (5202225.0 * P2z * P6x * P1y) + (9993375.0 * P2z * P4x * P3y) + (4380075.0 * P2z * P2x * P5y) - (411075.0 * P2z * P7y) - (141750.0 * P8x * P1y) - (411075.0 * P6x * P3y) - (382725.0 * P4x * P5y) - (99225.0 * P2x * P7y) + (14175.0 * P9y);
        const T gp6_term_5 = (10659600.0 * P6y * P2z * P1x) - (453600.0 * P8y * P1x) + (680400.0 * P6y * P3x) - (18002250.0 * P4y * P4z * P1x) - (17293500.0 * P4y * P2z * P3x) + (708750.0 * P4y * P5x) + (5202225.0 * P2y * P6z * P1x) + (9993375.0 * P2y * P4z * P3x) + (4380075.0 * P2y * P2z * P5x) - (411075.0 * P2y * P7x) - (141750.0 * P8z * P1x) - (411075.0 * P6z * P3x) - (382725.0 * P4z * P5x) - (99225.0 * P2z * P7x) + (14175.0 * P9x);
        const T gp6_term_6 = (10659600.0 * P6z * P2y * P1x) - (453600.0 * P8z * P1x) + (680400.0 * P6z * P3x) - (18002250.0 * P4z * P4y * P1x) - (17293500.0 * P4z * P2y * P3x) + (708750.0 * P4z * P5x) + (5202225.0 * P2z * P6y * P1x) + (9993375.0 * P2z * P4y * P3x) + (4380075.0 * P2z * P2y * P5x) - (411075.0 * P2z * P7x) - (141750.0 * P8y * P1x) - (411075.0 * P6y * P3x) - (382725.0 * P4y * P5x) - (99225.0 * P2y * P7x) + (14175.0 * P9x);

        const T gp6_miu_1 = temp * gp6_term_1;
        const T gp6_miu_2 = temp * gp6_term_2;
        const T gp6_miu_3 = temp * gp6_term_3;
        const T gp6_miu_4 = temp * gp6_term_4;
        const T gp6_miu_5 = temp * gp6_term_5;
        const T gp6_miu_6 = temp * gp6_term_6;

        value[24] += gp6_miu_1;
        value[25] += gp6_miu_2;
        value[26] += gp6_miu_3;
        value[27] += gp6_miu_4;
        value[28] += gp6_miu_5;
        value[29] += gp6_miu_6;


        // group 7
        const T gp7_term_1 = (19486005.0 * P5x * P4y) - (3412640.0 * P7x * P2y) + (5292210.0 * P5x * P2y * P2z) + (518980.0 * P9x) + (1576960.0 * P7x * P2z) + (2022405.0 * P5x * P4z) - (9524900.0 * P3x * P6y) - (1443750.0 * P3x * P4y * P2z) + (9471000.0 * P3x * P2y * P4z) + (1389850.0 * P3x * P6z) + (1516900.0 * P1x * P8y) + (2949100.0 * P1x * P6y * P2z) + (1772925.0 * P1x * P4y * P4z) + (766150.0 * P1x * P2y * P6z) + (425425.0 * P1x * P8z);
        const T gp7_term_2 = (19486005.0 * P5y * P4x) - (3412640.0 * P7y * P2x) + (5292210.0 * P5y * P2x * P2z) + (518980.0 * P9y) + (1576960.0 * P7y * P2z) + (2022405.0 * P5y * P4z) - (9524900.0 * P3y * P6x) - (1443750.0 * P3y * P4x * P2z) + (9471000.0 * P3y * P2x * P4z) + (1389850.0 * P3y * P6z) + (1516900.0 * P1y * P8x) + (2949100.0 * P1y * P6x * P2z) + (1772925.0 * P1y * P4x * P4z) + (766150.0 * P1y * P2x * P6z) + (425425.0 * P1y * P8z);
        const T gp7_term_3 = (19486005.0 * P5x * P4z) - (3412640.0 * P7x * P2z) + (5292210.0 * P5x * P2z * P2y) + (518980.0 * P9x) + (1576960.0 * P7x * P2y) + (2022405.0 * P5x * P4y) - (9524900.0 * P3x * P6z) - (1443750.0 * P3x * P4z * P2y) + (9471000.0 * P3x * P2z * P4y) + (1389850.0 * P3x * P6y) + (1516900.0 * P1x * P8z) + (2949100.0 * P1x * P6z * P2y) + (1772925.0 * P1x * P4z * P4y) + (766150.0 * P1x * P2z * P6y) + (425425.0 * P1x * P8y);
        const T gp7_term_4 = (19486005.0 * P5z * P4x) - (3412640.0 * P7z * P2x) + (5292210.0 * P5z * P2x * P2y) + (518980.0 * P9z) + (1576960.0 * P7z * P2y) + (2022405.0 * P5z * P4y) - (9524900.0 * P3z * P6x) - (1443750.0 * P3z * P4x * P2y) + (9471000.0 * P3z * P2x * P4y) + (1389850.0 * P3z * P6y) + (1516900.0 * P1z * P8x) + (2949100.0 * P1z * P6x * P2y) + (1772925.0 * P1z * P4x * P4y) + (766150.0 * P1z * P2x * P6y) + (425425.0 * P1z * P8y);
        const T gp7_term_5 = (19486005.0 * P5y * P4z) - (3412640.0 * P7y * P2z) + (5292210.0 * P5y * P2z * P2x) + (518980.0 * P9y) + (1576960.0 * P7y * P2x) + (2022405.0 * P5y * P4x) - (9524900.0 * P3y * P6z) - (1443750.0 * P3y * P4z * P2x) + (9471000.0 * P3y * P2z * P4x) + (1389850.0 * P3y * P6x) + (1516900.0 * P1y * P8z) + (2949100.0 * P1y * P6z * P2x) + (1772925.0 * P1y * P4z * P4x) + (766150.0 * P1y * P2z * P6x) + (425425.0 * P1y * P8x);
        const T gp7_term_6 = (19486005.0 * P5z * P4y) - (3412640.0 * P7z * P2y) + (5292210.0 * P5z * P2y * P2x) + (518980.0 * P9z) + (1576960.0 * P7z * P2x) + (2022405.0 * P5z * P4x) - (9524900.0 * P3z * P6y) - (1443750.0 * P3z * P4y * P2x) + (9471000.0 * P3z * P2y * P4x) + (1389850.0 * P3z * P6x) + (1516900.0 * P1z * P8y) + (2949100.0 * P1z * P6y * P2x) + (1772925.0 * P1z * P4y * P4x) + (766150.0 * P1z * P2y * P6x) + (425425.0 * P1z * P8x);

        const T gp7_miu_1 = temp * gp7_term_1;
        const T gp7_miu_2 = temp * gp7_term_2;
        const T gp7_miu_3 = temp * gp7_term_3;
        const T gp7_miu_4 = temp * gp7_term_4;
        const T gp7_miu_5 = temp * gp7_term_5;
        const T gp7_miu_6 = temp * gp7_term_6;

        value[30] += gp7_miu_1;
        value[31] += gp7_miu_2;
        value[32] += gp7_miu_3;
        value[33] += gp7_miu_4;
        value[34] += gp7_miu_5;
        value[35] += gp7_miu_6;


        // group 8
        const T gp8_term_1 = (16839900.0 * P5x * P3y * P1z) - (2494800.0 * P7x * P1y * P1z) + (623700.0 * P5x * P1y * P3z) - (13565475.0 * P3x * P5y * P1z) - (10914750.0 * P3x * P3y * P3z) + (2650725.0 * P3x * P1y * P5z) + (1559250.0 * P1x * P7y * P1z) + (2650725.0 * P1x * P5y * P3z) + (623700.0 * P1x * P3y * P5z) - (467775.0 * P1x * P1y * P7z);
        const T gp8_term_2 = (16839900.0 * P5y * P3x * P1z) - (2494800.0 * P7y * P1x * P1z) + (623700.0 * P5y * P1x * P3z) - (13565475.0 * P3y * P5x * P1z) - (10914750.0 * P3y * P3x * P3z) + (2650725.0 * P3y * P1x * P5z) + (1559250.0 * P1y * P7x * P1z) + (2650725.0 * P1y * P5x * P3z) + (623700.0 * P1y * P3x * P5z) - (467775.0 * P1y * P1x * P7z);
        const T gp8_term_3 = (16839900.0 * P5x * P3z * P1y) - (2494800.0 * P7x * P1z * P1y) + (623700.0 * P5x * P1z * P3y) - (13565475.0 * P3x * P5z * P1y) - (10914750.0 * P3x * P3z * P3y) + (2650725.0 * P3x * P1z * P5y) + (1559250.0 * P1x * P7z * P1y) + (2650725.0 * P1x * P5z * P3y) + (623700.0 * P1x * P3z * P5y) - (467775.0 * P1x * P1z * P7y);
        const T gp8_term_4 = (16839900.0 * P5z * P3x * P1y) - (2494800.0 * P7z * P1x * P1y) + (623700.0 * P5z * P1x * P3y) - (13565475.0 * P3z * P5x * P1y) - (10914750.0 * P3z * P3x * P3y) + (2650725.0 * P3z * P1x * P5y) + (1559250.0 * P1z * P7x * P1y) + (2650725.0 * P1z * P5x * P3y) + (623700.0 * P1z * P3x * P5y) - (467775.0 * P1z * P1x * P7y);
        const T gp8_term_5 = (16839900.0 * P5y * P3z * P1x) - (2494800.0 * P7y * P1z * P1x) + (623700.0 * P5y * P1z * P3x) - (13565475.0 * P3y * P5z * P1x) - (10914750.0 * P3y * P3z * P3x) + (2650725.0 * P3y * P1z * P5x) + (1559250.0 * P1y * P7z * P1x) + (2650725.0 * P1y * P5z * P3x) + (623700.0 * P1y * P3z * P5x) - (467775.0 * P1y * P1z * P7x);
        const T gp8_term_6 = (16839900.0 * P5z * P3y * P1x) - (2494800.0 * P7z * P1y * P1x) + (623700.0 * P5z * P1y * P3x) - (13565475.0 * P3z * P5y * P1x) - (10914750.0 * P3z * P3y * P3x) + (2650725.0 * P3z * P1y * P5x) + (1559250.0 * P1z * P7y * P1x) + (2650725.0 * P1z * P5y * P3x) + (623700.0 * P1z * P3y * P5x) - (467775.0 * P1z * P1y * P7x);

        const T gp8_miu_1 = temp * gp8_term_1;
        const T gp8_miu_2 = temp * gp8_term_2;
        const T gp8_miu_3 = temp * gp8_term_3;
        const T gp8_miu_4 = temp * gp8_term_4;
        const T gp8_miu_5 = temp * gp8_term_5;
        const T gp8_miu_6 = temp * gp8_term_6;

        value[36] += gp8_miu_1;
        value[37] += gp8_miu_2;
        value[38] += gp8_miu_3;
        value[39] += gp8_miu_4;
        value[40] += gp8_miu_5;
        value[41] += gp8_miu_6;


        // group 9
        const T gp9_term_1 = (16448670.0 * P5x * P2y * P2z) - (816480.0 * P7x * P2y) + (116235.0 * P5x * P4y) - (816480.0 * P7x * P2z) + (116235.0 * P5x * P4z) + (45360.0 * P9x) - (13707225.0 * P3x * P4y * P2z) - (13707225.0 * P3x * P2y * P4z) + (836325.0 * P3x * P6y) + (836325.0 * P3x * P6z) + (1460025.0 * P1x * P6y * P2z) + (3203550.0 * P1x * P4y * P4z) + (1460025.0 * P1x * P2y * P6z) - (141750.0 * P1x * P8y) - (141750.0 * P1x * P8z);
        const T gp9_term_2 = (16448670.0 * P5y * P2x * P2z) - (816480.0 * P7y * P2x) + (116235.0 * P5y * P4x) - (816480.0 * P7y * P2z) + (116235.0 * P5y * P4z) + (45360.0 * P9y) - (13707225.0 * P3y * P4x * P2z) - (13707225.0 * P3y * P2x * P4z) + (836325.0 * P3y * P6x) + (836325.0 * P3y * P6z) + (1460025.0 * P1y * P6x * P2z) + (3203550.0 * P1y * P4x * P4z) + (1460025.0 * P1y * P2x * P6z) - (141750.0 * P1y * P8x) - (141750.0 * P1y * P8z);
        const T gp9_term_3 = (16448670.0 * P5z * P2x * P2y) - (816480.0 * P7z * P2x) + (116235.0 * P5z * P4x) - (816480.0 * P7z * P2y) + (116235.0 * P5z * P4y) + (45360.0 * P9z) - (13707225.0 * P3z * P4x * P2y) - (13707225.0 * P3z * P2x * P4y) + (836325.0 * P3z * P6x) + (836325.0 * P3z * P6y) + (1460025.0 * P1z * P6x * P2y) + (3203550.0 * P1z * P4x * P4y) + (1460025.0 * P1z * P2x * P6y) - (141750.0 * P1z * P8x) - (141750.0 * P1z * P8y);

        const T gp9_miu_1 = temp * gp9_term_1;
        const T gp9_miu_2 = temp * gp9_term_2;
        const T gp9_miu_3 = temp * gp9_term_3;

        value[42] += gp9_miu_1;
        value[43] += gp9_miu_2;
        value[44] += gp9_miu_3;


        // group 10
        const T gp10_term_1 = (19604025.0 * P4x * P4y * P1z) - (7200900.0 * P6x * P2y * P1z) - (3203550.0 * P4x * P2y * P3z) + (226800.0 * P8x * P1z) + (283500.0 * P6x * P3z) - (104895.0 * P4x * P5z) - (7200900.0 * P2x * P6y * P1z) - (3203550.0 * P2x * P4y * P3z) + (3844260.0 * P2x * P2y * P5z) - (153090.0 * P2x * P7z) + (226800.0 * P8y * P1z) + (283500.0 * P6y * P3z) - (104895.0 * P4y * P5z) - (153090.0 * P2y * P7z) + (8505.0 * P9z);
        const T gp10_term_2 = (19604025.0 * P4x * P4z * P1y) - (7200900.0 * P6x * P2z * P1y) - (3203550.0 * P4x * P2z * P3y) + (226800.0 * P8x * P1y) + (283500.0 * P6x * P3y) - (104895.0 * P4x * P5y) - (7200900.0 * P2x * P6z * P1y) - (3203550.0 * P2x * P4z * P3y) + (3844260.0 * P2x * P2z * P5y) - (153090.0 * P2x * P7y) + (226800.0 * P8z * P1y) + (283500.0 * P6z * P3y) - (104895.0 * P4z * P5y) - (153090.0 * P2z * P7y) + (8505.0 * P9y);
        const T gp10_term_3 = (19604025.0 * P4y * P4z * P1x) - (7200900.0 * P6y * P2z * P1x) - (3203550.0 * P4y * P2z * P3x) + (226800.0 * P8y * P1x) + (283500.0 * P6y * P3x) - (104895.0 * P4y * P5x) - (7200900.0 * P2y * P6z * P1x) - (3203550.0 * P2y * P4z * P3x) + (3844260.0 * P2y * P2z * P5x) - (153090.0 * P2y * P7x) + (226800.0 * P8z * P1x) + (283500.0 * P6z * P3x) - (104895.0 * P4z * P5x) - (153090.0 * P2z * P7x) + (8505.0 * P9x);

        const T gp10_miu_1 = temp * gp10_term_1;
        const T gp10_miu_2 = temp * gp10_term_2;
        const T gp10_miu_3 = temp * gp10_term_3;

        value[45] += gp10_miu_1;
        value[46] += gp10_miu_2;
        value[47] += gp10_miu_3;


        // group 11
        const T gp11_term_1 = (20497050.0 * P4x * P3y * P2z) - (963900.0 * P6x * P3y) - (603855.0 * P4x * P5y) - (3458700.0 * P6x * P1y * P2z) - (1601775.0 * P4x * P1y * P4z) + (226800.0 * P8x * P1y) - (8224335.0 * P2x * P5y * P2z) - (6789825.0 * P2x * P3y * P4z) + (564165.0 * P2x * P7y) + (1998675.0 * P2x * P1y * P6z) + (252315.0 * P7y * P2z) + (487620.0 * P5y * P4z) + (127575.0 * P3y * P6z) - (22680.0 * P9y) - (85050.0 * P1y * P8z);
        const T gp11_term_2 = (20497050.0 * P4y * P3x * P2z) - (963900.0 * P6y * P3x) - (603855.0 * P4y * P5x) - (3458700.0 * P6y * P1x * P2z) - (1601775.0 * P4y * P1x * P4z) + (226800.0 * P8y * P1x) - (8224335.0 * P2y * P5x * P2z) - (6789825.0 * P2y * P3x * P4z) + (564165.0 * P2y * P7x) + (1998675.0 * P2y * P1x * P6z) + (252315.0 * P7x * P2z) + (487620.0 * P5x * P4z) + (127575.0 * P3x * P6z) - (22680.0 * P9x) - (85050.0 * P1x * P8z);
        const T gp11_term_3 = (20497050.0 * P4x * P3z * P2y) - (963900.0 * P6x * P3z) - (603855.0 * P4x * P5z) - (3458700.0 * P6x * P1z * P2y) - (1601775.0 * P4x * P1z * P4y) + (226800.0 * P8x * P1z) - (8224335.0 * P2x * P5z * P2y) - (6789825.0 * P2x * P3z * P4y) + (564165.0 * P2x * P7z) + (1998675.0 * P2x * P1z * P6y) + (252315.0 * P7z * P2y) + (487620.0 * P5z * P4y) + (127575.0 * P3z * P6y) - (22680.0 * P9z) - (85050.0 * P1z * P8y);
        const T gp11_term_4 = (20497050.0 * P4z * P3x * P2y) - (963900.0 * P6z * P3x) - (603855.0 * P4z * P5x) - (3458700.0 * P6z * P1x * P2y) - (1601775.0 * P4z * P1x * P4y) + (226800.0 * P8z * P1x) - (8224335.0 * P2z * P5x * P2y) - (6789825.0 * P2z * P3x * P4y) + (564165.0 * P2z * P7x) + (1998675.0 * P2z * P1x * P6y) + (252315.0 * P7x * P2y) + (487620.0 * P5x * P4y) + (127575.0 * P3x * P6y) - (22680.0 * P9x) - (85050.0 * P1x * P8y);
        const T gp11_term_5 = (20497050.0 * P4y * P3z * P2x) - (963900.0 * P6y * P3z) - (603855.0 * P4y * P5z) - (3458700.0 * P6y * P1z * P2x) - (1601775.0 * P4y * P1z * P4x) + (226800.0 * P8y * P1z) - (8224335.0 * P2y * P5z * P2x) - (6789825.0 * P2y * P3z * P4x) + (564165.0 * P2y * P7z) + (1998675.0 * P2y * P1z * P6x) + (252315.0 * P7z * P2x) + (487620.0 * P5z * P4x) + (127575.0 * P3z * P6x) - (22680.0 * P9z) - (85050.0 * P1z * P8x);
        const T gp11_term_6 = (20497050.0 * P4z * P3y * P2x) - (963900.0 * P6z * P3y) - (603855.0 * P4z * P5y) - (3458700.0 * P6z * P1y * P2x) - (1601775.0 * P4z * P1y * P4x) + (226800.0 * P8z * P1y) - (8224335.0 * P2z * P5y * P2x) - (6789825.0 * P2z * P3y * P4x) + (564165.0 * P2z * P7y) + (1998675.0 * P2z * P1y * P6x) + (252315.0 * P7y * P2x) + (487620.0 * P5y * P4x) + (127575.0 * P3y * P6x) - (22680.0 * P9y) - (85050.0 * P1y * P8x);

        const T gp11_miu_1 = temp * gp11_term_1;
        const T gp11_miu_2 = temp * gp11_term_2;
        const T gp11_miu_3 = temp * gp11_term_3;
        const T gp11_miu_4 = temp * gp11_term_4;
        const T gp11_miu_5 = temp * gp11_term_5;
        const T gp11_miu_6 = temp * gp11_term_6;

        value[48] += gp11_miu_1;
        value[49] += gp11_miu_2;
        value[50] += gp11_miu_3;
        value[51] += gp11_miu_4;
        value[52] += gp11_miu_5;
        value[53] += gp11_miu_6;

        // group 12
        const T gp12_term = (21829320.0 * P3x * P3y * P3z) - (3274515.0 * P5x * P3y * P1z) - (3274605.0 * P3x * P5y * P1z) - (3274425.0 * P5x * P1y * P3z) - (3274425.0 * P3x * P1y * P5z) + (935550.0 * P7x * P1y * P1z) - (3274605.0 * P1x * P5y * P3z) - (3274515.0 * P1x * P3y * P5z) + (935460.0 * P1x * P7y * P1z) + (935550.0 * P1x * P1y * P7z);
        const T gp12_m = temp * gp12_term;
        value[54] += gp12_m;
    }
}}