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
        const array3d_t<T> lambda_dr = dr * lambda;
        const array3d_t<T> lambda_dr2 = lambda_dr * lambda_dr;
        const array3d_t<T> lambda_dr3 = lambda_dr2 * lambda_dr;
        const array3d_t<T> lambda_dr4 = lambda_dr3 * lambda_dr;
        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;

        const array3d_t<T> P1_vals = lambda_dr;
        array3d_t<T> P2_vals;
        array3d_t<T> P3_vals;
        array3d_t<T> P4_vals;

        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            P2_vals[dim] = P2(lambda_dr2[dim], inv_gamma);
            P3_vals[dim] = P3(lambda_dr[dim], lambda_dr3[dim], inv_gamma);
            P4_vals[dim] = P4(lambda_dr2[dim], lambda_dr4[dim], inv_gamma, inv_gamma_2);
        }

        // group 1: (24*P4[i]) + 9*(sum_P4 - P4[i]) - 72*P2[i]*(sum_P2 - P2[i]) + 18*P2[j]*P2[k]
        // where j and k are the other two dimensions
        const T sum_P4 = P4_vals[0] + P4_vals[1] + P4_vals[2];
        const T sum_P2 = P2_vals[0] + P2_vals[1] + P2_vals[2];
        
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp1_term = (24.0 * P4_vals[dim]) + (9.0 * (sum_P4 - P4_vals[dim])) 
                             - (72.0 * P2_vals[dim] * (sum_P2 - P2_vals[dim])) 
                             + (18.0 * P2_vals[dim1] * P2_vals[dim2]);
            add_to_atomic(value, base_key + dim, temp * gp1_term, hash_table, local_tid);
        }

        // group 2: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        constexpr int pairs[6][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp2_term = (60.0 * P3_vals[a] * P1_vals[b]) 
                             - (45.0 * (P1_vals[a] * P3_vals[b] + P1_vals[a] * P1_vals[b] * P2_vals[c]));
            add_to_atomic(value, base_key + 3 + pair, temp * gp2_term, hash_table, local_tid);
        }

        // group 3: 3 terms for pairs (x,y), (x,z), (y,z)
        constexpr int pairs2[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        #pragma unroll
        for (int pair = 0; pair < 3; ++pair) {
            const int a = pairs2[pair][0];
            const int b = pairs2[pair][1];
            const int c = 3 - a - b;
            const T gp3_term = (81.0 * P2_vals[a] * P2_vals[b]) 
                             - (12.0 * (P4_vals[a] + P4_vals[b])) 
                             + (3.0 * P4_vals[c]) 
                             - (9.0 * P2_vals[c] * (P2_vals[a] + P2_vals[b]));
            add_to_atomic(value, base_key + 9 + pair, temp * gp3_term, hash_table, local_tid);
        }

        // group 4: 3 terms for (x,y,z), (y,x,z), (z,x,y) - but actually cyclic permutations
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int a = dim;
            const int b = (dim + 1) % 3;
            const int c = (dim + 2) % 3;
            const T gp4_term = (90.0 * P2_vals[a] * P1_vals[b] * P1_vals[c]) 
                             - (15.0 * (P3_vals[b] * P1_vals[c] + P1_vals[b] * P3_vals[c]));
            add_to_atomic(value, base_key + 12 + dim, temp * gp4_term, hash_table, local_tid);
        }
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_5(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const array3d_t<T> lambda_dr = dr * lambda;
        const array3d_t<T> lambda_dr2 = lambda_dr * lambda_dr;
        const array3d_t<T> lambda_dr3 = lambda_dr2 * lambda_dr;
        const array3d_t<T> lambda_dr4 = lambda_dr3 * lambda_dr;
        const array3d_t<T> lambda_dr5 = lambda_dr4 * lambda_dr;
        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;

        const array3d_t<T> P1_vals = lambda_dr;
        array3d_t<T> P2_vals;
        array3d_t<T> P3_vals;
        array3d_t<T> P4_vals;
        array3d_t<T> P5_vals;

        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            P2_vals[dim] = P2(lambda_dr2[dim], inv_gamma);
            P3_vals[dim] = P3(lambda_dr[dim], lambda_dr3[dim], inv_gamma);
            P4_vals[dim] = P4(lambda_dr2[dim], lambda_dr4[dim], inv_gamma, inv_gamma_2);
            P5_vals[dim] = P5(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], inv_gamma, inv_gamma_2);
        }

        // group 1: (120*P5[i]) - 600*P3[i]*(sum_P2 - P2[i]) + 225*P1[i]*(sum_P4 - P4[i]) + 450*P1[i]*P2[j]*P2[k]
        const T sum_P2 = P2_vals[0] + P2_vals[1] + P2_vals[2];
        const T sum_P4 = P4_vals[0] + P4_vals[1] + P4_vals[2];
        
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp1_term = (120.0 * P5_vals[dim]) 
                             - (600.0 * P3_vals[dim] * (sum_P2 - P2_vals[dim])) 
                             + (225.0 * P1_vals[dim] * (sum_P4 - P4_vals[dim])) 
                             + (450.0 * P1_vals[dim] * P2_vals[dim1] * P2_vals[dim2]);
            add_to_atomic(value, base_key + dim, temp * gp1_term, hash_table, local_tid);
        }

        // group 2: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        constexpr int pairs[6][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp2_term = (360.0 * P4_vals[a] * P1_vals[b]) 
                             - (540.0 * P2_vals[a] * (P3_vals[b] + P1_vals[b] * P2_vals[c])) 
                             + (45.0 * (P5_vals[b] + P1_vals[b] * P4_vals[c])) 
                             + (90.0 * P3_vals[b] * P2_vals[c]);
            add_to_atomic(value, base_key + 3 + pair, temp * gp2_term, hash_table, local_tid);
        }

        // group 3: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp3_term = (615.0 * P3_vals[a] * P2_vals[b]) 
                             - (60.0 * P5_vals[a]) 
                             - (15.0 * P3_vals[a] * P2_vals[c]) 
                             - (270.0 * P1_vals[a] * P4_vals[b]) 
                             + (45.0 * P1_vals[a] * P4_vals[c]) 
                             - (225.0 * P1_vals[a] * P2_vals[b] * P2_vals[c]);
            add_to_atomic(value, base_key + 9 + pair, temp * gp3_term, hash_table, local_tid);
        }

        // group 4: 3 terms for cyclic permutations
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int a = dim;
            const int b = (dim + 1) % 3;
            const int c = (dim + 2) % 3;
            const T gp4_term = (630.0 * P3_vals[a] * P1_vals[b] * P1_vals[c]) 
                             - (315.0 * P1_vals[a] * (P3_vals[b] * P1_vals[c] + P1_vals[b] * P3_vals[c]));
            add_to_atomic(value, base_key + 15 + dim, temp * gp4_term, hash_table, local_tid);
        }

        // group 5: 3 terms for pairs (x,y), (x,z), (y,z)
        constexpr int pairs2[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        #pragma unroll
        for (int pair = 0; pair < 3; ++pair) {
            const int a = pairs2[pair][0];
            const int b = pairs2[pair][1];
            const int c = 3 - a - b;
            const T gp5_term = (765.0 * P2_vals[a] * P2_vals[b] * P1_vals[c]) 
                             - (90.0 * P1_vals[c] * (P4_vals[a] + P4_vals[b])) 
                             - (75.0 * P3_vals[c] * (P2_vals[a] + P2_vals[b])) 
                             + (15.0 * P5_vals[c]);
            add_to_atomic(value, base_key + 18 + pair, temp * gp5_term, hash_table, local_tid);
        }
    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_6(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const array3d_t<T> lambda_dr = dr * lambda;
        const array3d_t<T> lambda_dr2 = lambda_dr * lambda_dr;
        const array3d_t<T> lambda_dr3 = lambda_dr2 * lambda_dr;
        const array3d_t<T> lambda_dr4 = lambda_dr3 * lambda_dr;
        const array3d_t<T> lambda_dr5 = lambda_dr4 * lambda_dr;
        const array3d_t<T> lambda_dr6 = lambda_dr5 * lambda_dr;
        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;
        const T inv_gamma_3 = inv_gamma_2 * inv_gamma;

        const array3d_t<T> P1_vals = lambda_dr;
        array3d_t<T> P2_vals;
        array3d_t<T> P3_vals;
        array3d_t<T> P4_vals;
        array3d_t<T> P5_vals;
        array3d_t<T> P6_vals;

        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            P2_vals[dim] = P2(lambda_dr2[dim], inv_gamma);
            P3_vals[dim] = P3(lambda_dr[dim], lambda_dr3[dim], inv_gamma);
            P4_vals[dim] = P4(lambda_dr2[dim], lambda_dr4[dim], inv_gamma, inv_gamma_2);
            P5_vals[dim] = P5(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], inv_gamma, inv_gamma_2);
            P6_vals[dim] = P6(lambda_dr2[dim], lambda_dr4[dim], lambda_dr6[dim], inv_gamma, inv_gamma_2, inv_gamma_3);
        }

        // group 1: 3 terms for x, y, z
        // Compute in original order to match floating-point precision
        const T P2_product = P2_vals[0] * P2_vals[1] * P2_vals[2];
        const T sum_P6 = P6_vals[0] + P6_vals[1] + P6_vals[2];
        
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            // Match original operation order more closely to preserve floating-point precision
            const T gp1_term = (720.0 * P6_vals[dim]) 
                             - (5400.0 * P4_vals[dim] * P2_vals[dim1]) 
                             - (5400.0 * P4_vals[dim] * P2_vals[dim2]) 
                             + (4050.0 * P2_vals[dim] * P4_vals[dim1]) 
                             + (8100.0 * P2_vals[dim] * P2_vals[dim1] * P2_vals[dim2]) 
                             + (4050.0 * P2_vals[dim] * P4_vals[dim2]) 
                             - (225.0 * P6_vals[dim1]) 
                             - (675.0 * P4_vals[dim1] * P2_vals[dim2]) 
                             - (675.0 * P2_vals[dim1] * P4_vals[dim2]) 
                             - (225.0 * P6_vals[dim2]);
            add_to_atomic(value, base_key + dim, temp * gp1_term, hash_table, local_tid);
        }


        // group 2: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        constexpr int pairs[6][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp2_term = (2520.0 * P5_vals[a] * P1_vals[b]) 
                             - (6300.0 * P3_vals[a] * P3_vals[b]) 
                             - (6300.0 * P3_vals[a] * P1_vals[b] * P2_vals[c]) 
                             + (1575.0 * P1_vals[a] * P5_vals[b]) 
                             + (3150.0 * P1_vals[a] * P3_vals[b] * P2_vals[c]) 
                             + (1575.0 * P1_vals[a] * P1_vals[b] * P4_vals[c]);
            add_to_atomic(value, base_key + 3 + pair, temp * gp2_term, hash_table, local_tid);
        }

        // group 3: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp3_term = (5220.0 * P4_vals[a] * P2_vals[b]) 
                             - (360.0 * P6_vals[a]) 
                             + (180.0 * P4_vals[a] * P2_vals[c]) 
                             - (4545.0 * P2_vals[a] * P4_vals[b]) 
                             - (4050.0 * P2_vals[a] * P2_vals[b] * P2_vals[c]) 
                             + (495.0 * P2_vals[a] * P4_vals[c]) 
                             + (270.0 * P6_vals[b]) 
                             + (495.0 * P4_vals[b] * P2_vals[c]) 
                             + (180.0 * P2_vals[b] * P4_vals[c]) 
                             - (45.0 * P6_vals[c]);
            add_to_atomic(value, base_key + 9 + pair, temp * gp3_term, hash_table, local_tid);
        }

        // group 4: 3 terms for cyclic permutations
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int a = dim;
            const int b = (dim + 1) % 3;
            const int c = (dim + 2) % 3;
            const T gp4_term = (5040.0 * P4_vals[a] * P1_vals[b] * P1_vals[c]) 
                             - (5040.0 * P2_vals[a] * P3_vals[b] * P1_vals[c]) 
                             - (5040.0 * P2_vals[a] * P1_vals[b] * P3_vals[c]) 
                             + (315.0 * P5_vals[b] * P1_vals[c]) 
                             + (630.0 * P3_vals[b] * P3_vals[c]) 
                             + (315.0 * P1_vals[b] * P5_vals[c]);
            add_to_atomic(value, base_key + 15 + dim, temp * gp4_term, hash_table, local_tid);
        }

        // group 5: 3 terms for pairs (x,y), (x,z), (y,z)
        constexpr int pairs2[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        #pragma unroll
        for (int pair = 0; pair < 3; ++pair) {
            const int a = pairs2[pair][0];
            const int b = pairs2[pair][1];
            const int c = 3 - a - b;
            const T gp5_term = (6615.0 * P3_vals[a] * P3_vals[b]) 
                             - (1890.0 * P5_vals[a] * P1_vals[b]) 
                             - (945.0 * P3_vals[a] * P1_vals[b] * P2_vals[c]) 
                             - (1890.0 * P1_vals[a] * P5_vals[b]) 
                             - (945.0 * P1_vals[a] * P3_vals[b] * P2_vals[c]) 
                             + (945.0 * P1_vals[a] * P1_vals[b] * P4_vals[c]);
            add_to_atomic(value, base_key + 18 + pair, temp * gp5_term, hash_table, local_tid);
        }

        // group 6: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp6_term = (7245.0 * P3_vals[a] * P2_vals[b] * P1_vals[c]) 
                             - (630.0 * P5_vals[a] * P1_vals[c]) 
                             - (315.0 * P3_vals[a] * P3_vals[c]) 
                             - (2520.0 * P1_vals[a] * P4_vals[b] * P1_vals[c]) 
                             - (2205.0 * P1_vals[a] * P2_vals[b] * P3_vals[c]) 
                             + (315.0 * P1_vals[a] * P5_vals[c]);
            add_to_atomic(value, base_key + 21 + pair, temp * gp6_term, hash_table, local_tid);
        }


        // group 7: single term
        const T gp7_term = (8100.0 * P2_product) 
                         - (675.0 * (P4_vals[0] * P2_vals[1] + P2_vals[0] * P4_vals[1] 
                                   + P4_vals[0] * P2_vals[2] + P2_vals[0] * P4_vals[2] 
                                   + P4_vals[1] * P2_vals[2] + P2_vals[1] * P4_vals[2])) 
                         + (90.0 * sum_P6);
        add_to_atomic(value, base_key + 27, temp * gp7_term, hash_table, local_tid);

    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_7(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const array3d_t<T> lambda_dr = dr * lambda;
        const array3d_t<T> lambda_dr2 = lambda_dr * lambda_dr;
        const array3d_t<T> lambda_dr3 = lambda_dr2 * lambda_dr;
        const array3d_t<T> lambda_dr4 = lambda_dr3 * lambda_dr;
        const array3d_t<T> lambda_dr5 = lambda_dr4 * lambda_dr;
        const array3d_t<T> lambda_dr6 = lambda_dr5 * lambda_dr;
        const array3d_t<T> lambda_dr7 = lambda_dr6 * lambda_dr;
        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;
        const T inv_gamma_3 = inv_gamma_2 * inv_gamma;

        const array3d_t<T> P1_vals = lambda_dr;
        array3d_t<T> P2_vals;
        array3d_t<T> P3_vals;
        array3d_t<T> P4_vals;
        array3d_t<T> P5_vals;
        array3d_t<T> P6_vals;
        array3d_t<T> P7_vals;

        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            P2_vals[dim] = P2(lambda_dr2[dim], inv_gamma);
            P3_vals[dim] = P3(lambda_dr[dim], lambda_dr3[dim], inv_gamma);
            P4_vals[dim] = P4(lambda_dr2[dim], lambda_dr4[dim], inv_gamma, inv_gamma_2);
            P5_vals[dim] = P5(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], inv_gamma, inv_gamma_2);
            P6_vals[dim] = P6(lambda_dr2[dim], lambda_dr4[dim], lambda_dr6[dim], inv_gamma, inv_gamma_2, inv_gamma_3);
            P7_vals[dim] = P7(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], lambda_dr7[dim], inv_gamma, inv_gamma_2, inv_gamma_3);
        }

        // group 1: 3 terms for x, y, z
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp1_term = (5040.0 * P7_vals[dim]) 
                             - (52920.0 * P5_vals[dim] * P2_vals[dim1]) 
                             - (52920.0 * P5_vals[dim] * P2_vals[dim2]) 
                             + (66150.0 * P3_vals[dim] * P4_vals[dim1]) 
                             + (132300.0 * P3_vals[dim] * P2_vals[dim1] * P2_vals[dim2]) 
                             + (66150.0 * P3_vals[dim] * P4_vals[dim2]) 
                             - (11025.0 * P1_vals[dim] * P6_vals[dim1]) 
                             - (33075.0 * P1_vals[dim] * P4_vals[dim1] * P2_vals[dim2]) 
                             - (33075.0 * P1_vals[dim] * P2_vals[dim1] * P4_vals[dim2]) 
                             - (11025.0 * P1_vals[dim] * P6_vals[dim2]);
            add_to_atomic(value, base_key + dim, temp * gp1_term, hash_table, local_tid);
        }

        // group 2: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        constexpr int pairs[6][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp2_term = (20160.0 * P6_vals[a] * P1_vals[b]) 
                             - (75600.0 * P4_vals[a] * P3_vals[b]) 
                             - (75600.0 * P4_vals[a] * P1_vals[b] * P2_vals[c]) 
                             + (37800.0 * P2_vals[a] * P5_vals[b]) 
                             + (75600.0 * P2_vals[a] * P3_vals[b] * P2_vals[c]) 
                             + (37800.0 * P2_vals[a] * P1_vals[b] * P4_vals[c]) 
                             - (1575.0 * P7_vals[b]) 
                             - (4725.0 * P5_vals[b] * P2_vals[c]) 
                             - (4725.0 * P3_vals[b] * P4_vals[c]) 
                             - (1575.0 * P1_vals[b] * P6_vals[c]);
            add_to_atomic(value, base_key + 3 + pair, temp * gp2_term, hash_table, local_tid);
        }

        // group 3: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp3_term = (49140.0 * P5_vals[a] * P2_vals[b]) 
                             - (2520.0 * P7_vals[a]) 
                             + (3780.0 * P5_vals[a] * P2_vals[c]) 
                             - (70875.0 * P3_vals[a] * P4_vals[b]) 
                             - (66150.0 * P3_vals[a] * P2_vals[b] * P2_vals[c]) 
                             + (4725.0 * P3_vals[a] * P4_vals[c]) 
                             + (12600.0 * P1_vals[a] * P6_vals[b]) 
                             + (23625.0 * P1_vals[a] * P4_vals[b] * P2_vals[c]) 
                             + (9450.0 * P1_vals[a] * P2_vals[b] * P4_vals[c]) 
                             - (1575.0 * P1_vals[a] * P6_vals[c]);
            add_to_atomic(value, base_key + 9 + pair, temp * gp3_term, hash_table, local_tid);
        }


        // group 4: 3 terms for cyclic permutations
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int a = dim;
            const int b = (dim + 1) % 3;
            const int c = (dim + 2) % 3;
            const T gp4_term = (45360.0 * P5_vals[a] * P1_vals[b] * P1_vals[c]) 
                             - (75600.0 * P3_vals[a] * P3_vals[b] * P1_vals[c]) 
                             - (75600.0 * P3_vals[a] * P1_vals[b] * P3_vals[c]) 
                             + (14175.0 * P1_vals[a] * P5_vals[b] * P1_vals[c]) 
                             + (28350.0 * P1_vals[a] * P3_vals[b] * P3_vals[c]) 
                             + (14175.0 * P1_vals[a] * P1_vals[b] * P5_vals[c]);
            add_to_atomic(value, base_key + 15 + dim, temp * gp4_term, hash_table, local_tid);
        }


        // group 5: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp5_term = (75600.0 * P4_vals[a] * P3_vals[b]) 
                             - (15120.0 * P6_vals[a] * P1_vals[b]) 
                             - (42525.0 * P2_vals[a] * P5_vals[b]) 
                             - (28350.0 * P2_vals[a] * P3_vals[b] * P2_vals[c]) 
                             + (14175.0 * P2_vals[a] * P1_vals[b] * P4_vals[c]) 
                             + (1890.0 * P7_vals[b]) 
                             + (2835.0 * P5_vals[b] * P2_vals[c]) 
                             - (945.0 * P1_vals[b] * P6_vals[c]);
            add_to_atomic(value, base_key + 18 + pair, temp * gp5_term, hash_table, local_tid);
        }


        // group 6: 6 terms for pairs (x,y), (y,x), (x,z), (z,x), (y,z), (z,y)
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp6_term = (75600.0 * P4_vals[a] * P2_vals[b] * P1_vals[c]) 
                             - (5040.0 * P6_vals[a] * P1_vals[c]) 
                             - (51975.0 * P2_vals[a] * P4_vals[b] * P1_vals[c]) 
                             - (47250.0 * P2_vals[a] * P2_vals[b] * P3_vals[c]) 
                             + (4725.0 * P2_vals[a] * P5_vals[c]) 
                             + (2520.0 * P6_vals[b] * P1_vals[c]) 
                             + (4725.0 * P4_vals[b] * P3_vals[c]) 
                             + (1890.0 * P2_vals[b] * P5_vals[c]) 
                             - (315.0 * P7_vals[c]);
            add_to_atomic(value, base_key + 24 + pair, temp * gp6_term, hash_table, local_tid);
        }


        // group 7: 3 terms for cyclic permutations
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int a = dim;
            const int b = (dim + 1) % 3;
            const int c = (dim + 2) % 3;
            const T gp7_term = (89775.0 * P3_vals[a] * P3_vals[b] * P1_vals[c]) 
                             - (22680.0 * P5_vals[a] * P1_vals[b] * P1_vals[c]) 
                             - (14175.0 * P3_vals[a] * P1_vals[b] * P3_vals[c]) 
                             - (22680.0 * P1_vals[a] * P5_vals[b] * P1_vals[c]) 
                             - (14175.0 * P1_vals[a] * P3_vals[b] * P3_vals[c]) 
                             + (8505.0 * P1_vals[a] * P1_vals[b] * P5_vals[c]);
            add_to_atomic(value, base_key + 30 + dim, temp * gp7_term, hash_table, local_tid);
        }


        // group 8: 3 terms for x, y, z
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp8_term = (94500.0 * P3_vals[dim] * P2_vals[dim1] * P2_vals[dim2]) 
                             - (6615.0 * P5_vals[dim] * P2_vals[dim2]) 
                             - (4725.0 * P3_vals[dim] * P4_vals[dim2]) 
                             - (6615.0 * P5_vals[dim] * P2_vals[dim1]) 
                             - (4725.0 * P3_vals[dim] * P4_vals[dim1]) 
                             + (630.0 * P7_vals[dim]) 
                             - (23625.0 * P1_vals[dim] * P4_vals[dim1] * P2_vals[dim2]) 
                             - (23625.0 * P1_vals[dim] * P2_vals[dim1] * P4_vals[dim2]) 
                             + (2520.0 * P1_vals[dim] * P6_vals[dim2]) 
                             + (2520.0 * P1_vals[dim] * P6_vals[dim1]);
            add_to_atomic(value, base_key + 33 + dim, temp * gp8_term, hash_table, local_tid);
        }

    }
    
    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_8(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const array3d_t<T> lambda_dr = dr * lambda;
        const array3d_t<T> lambda_dr2 = lambda_dr * lambda_dr;
        const array3d_t<T> lambda_dr3 = lambda_dr2 * lambda_dr;
        const array3d_t<T> lambda_dr4 = lambda_dr3 * lambda_dr;
        const array3d_t<T> lambda_dr5 = lambda_dr4 * lambda_dr;
        const array3d_t<T> lambda_dr6 = lambda_dr5 * lambda_dr;
        const array3d_t<T> lambda_dr7 = lambda_dr6 * lambda_dr;
        const array3d_t<T> lambda_dr8 = lambda_dr7 * lambda_dr;
        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;
        const T inv_gamma_3 = inv_gamma_2 * inv_gamma;
        const T inv_gamma_4 = inv_gamma_3 * inv_gamma;

        const array3d_t<T> P1_vals = lambda_dr;
        array3d_t<T> P2_vals;
        array3d_t<T> P3_vals;
        array3d_t<T> P4_vals;
        array3d_t<T> P5_vals;
        array3d_t<T> P6_vals;
        array3d_t<T> P7_vals;
        array3d_t<T> P8_vals;

        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            P2_vals[dim] = P2(lambda_dr2[dim], inv_gamma);
            P3_vals[dim] = P3(lambda_dr[dim], lambda_dr3[dim], inv_gamma);
            P4_vals[dim] = P4(lambda_dr2[dim], lambda_dr4[dim], inv_gamma, inv_gamma_2);
            P5_vals[dim] = P5(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], inv_gamma, inv_gamma_2);
            P6_vals[dim] = P6(lambda_dr2[dim], lambda_dr4[dim], lambda_dr6[dim], inv_gamma, inv_gamma_2, inv_gamma_3);
            P7_vals[dim] = P7(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], lambda_dr7[dim], inv_gamma, inv_gamma_2, inv_gamma_3);
            P8_vals[dim] = P8(lambda_dr2[dim], lambda_dr4[dim], lambda_dr6[dim], lambda_dr8[dim], inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
        }

        // group 1: 3 terms for x, y, z
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp1_term = (40320.0 * P8_vals[dim]) - (564480.0 * P6_vals[dim] * P2_vals[dim1]) - (564480.0 * P6_vals[dim] * P2_vals[dim2]) 
                             + (1058400.0 * P4_vals[dim] * P4_vals[dim1]) + (2116800.0 * P4_vals[dim] * P2_vals[dim1] * P2_vals[dim2]) 
                             + (1058400.0 * P4_vals[dim] * P4_vals[dim2]) - (352800.0 * P2_vals[dim] * P6_vals[dim1]) 
                             - (1058400.0 * P2_vals[dim] * P4_vals[dim1] * P2_vals[dim2]) - (1058400.0 * P2_vals[dim] * P2_vals[dim1] * P4_vals[dim2]) 
                             - (352800.0 * P2_vals[dim] * P6_vals[dim2]) + (11025.0 * P8_vals[dim1]) + (44100.0 * P6_vals[dim1] * P2_vals[dim2]) 
                             + (66150.0 * P4_vals[dim1] * P4_vals[dim2]) + (44100.0 * P2_vals[dim1] * P6_vals[dim2]) + (11025.0 * P8_vals[dim2]);
            add_to_atomic(value, base_key + dim, temp * gp1_term, hash_table, local_tid);
        }

        // group 2: 6 terms for pairs
        constexpr int pairs[6][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp2_term = (181440.0 * P7_vals[a] * P1_vals[b]) - (952560.0 * P5_vals[a] * P3_vals[b]) 
                             - (952560.0 * P5_vals[a] * P1_vals[b] * P2_vals[c]) + (793800.0 * P3_vals[a] * P5_vals[b]) 
                             + (1587600.0 * P3_vals[a] * P3_vals[b] * P2_vals[c]) + (793800.0 * P3_vals[a] * P1_vals[b] * P4_vals[c]) 
                             - (99225.0 * P1_vals[a] * P7_vals[b]) - (297675.0 * P1_vals[a] * P5_vals[b] * P2_vals[c]) 
                             - (297675.0 * P1_vals[a] * P3_vals[b] * P4_vals[c]) - (99225.0 * P1_vals[a] * P1_vals[b] * P6_vals[c]);
            add_to_atomic(value, base_key + 3 + pair, temp * gp2_term, hash_table, local_tid);
        }


        // group 3: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp3_term = (509040.0 * P6_vals[a] * P2_vals[b]) - (20160.0 * P8_vals[a]) + (55440.0 * P6_vals[a] * P2_vals[c]) 
                             - (1096200.0 * P4_vals[a] * P4_vals[b]) - (1058400.0 * P4_vals[a] * P2_vals[b] * P2_vals[c]) 
                             + (37800.0 * P4_vals[a] * P4_vals[c]) + (389025.0 * P2_vals[a] * P6_vals[b]) 
                             + (741825.0 * P2_vals[a] * P4_vals[b] * P2_vals[c]) + (316575.0 * P2_vals[a] * P2_vals[b] * P4_vals[c]) 
                             - (36225.0 * P2_vals[a] * P6_vals[c]) - (12600.0 * P8_vals[b]) - (36225.0 * P6_vals[b] * P2_vals[c]) 
                             - (33075.0 * P4_vals[b] * P4_vals[c]) - (7875.0 * P2_vals[b] * P6_vals[c]) + (1575.0 * P8_vals[c]);
            add_to_atomic(value, base_key + 9 + pair, temp * gp3_term, hash_table, local_tid);
        }


        // group 4: 3 terms for x, y, z
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp4_term = (453600.0 * P6_vals[dim] * P1_vals[dim1] * P1_vals[dim2]) 
                             - (1134000.0 * P4_vals[dim] * P3_vals[dim1] * P1_vals[dim2]) 
                             - (1134000.0 * P4_vals[dim] * P1_vals[dim1] * P3_vals[dim2]) 
                             + (425250.0 * P2_vals[dim] * P5_vals[dim1] * P1_vals[dim2]) 
                             + (850500.0 * P2_vals[dim] * P3_vals[dim1] * P3_vals[dim2]) 
                             + (425250.0 * P2_vals[dim] * P1_vals[dim1] * P5_vals[dim2]) 
                             - (14175.0 * P7_vals[dim1] * P1_vals[dim2]) - (42525.0 * P5_vals[dim1] * P3_vals[dim2]) 
                             - (42525.0 * P3_vals[dim1] * P5_vals[dim2]) - (14175.0 * P1_vals[dim1] * P7_vals[dim2]);
            add_to_atomic(value, base_key + 15 + dim, temp * gp4_term, hash_table, local_tid);
        }


        // group 5: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp5_term = (922320.0 * P5_vals[a] * P3_vals[b]) - (136080.0 * P7_vals[a] * P1_vals[b]) 
                             + (90720.0 * P5_vals[a] * P1_vals[b] * P2_vals[c]) - (855225.0 * P3_vals[a] * P5_vals[b]) 
                             - (670950.0 * P3_vals[a] * P3_vals[b] * P2_vals[c]) + (184275.0 * P3_vals[a] * P1_vals[b] * P4_vals[c]) 
                             + (113400.0 * P1_vals[a] * P7_vals[b]) + (184275.0 * P1_vals[a] * P5_vals[b] * P2_vals[c]) 
                             + (28350.0 * P1_vals[a] * P3_vals[b] * P4_vals[c]) - (42525.0 * P1_vals[a] * P1_vals[b] * P6_vals[c]);
            add_to_atomic(value, base_key + 18 + pair, temp * gp5_term, hash_table, local_tid);
        }


        // group 6: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp6_term = (861840.0 * P5_vals[a] * P2_vals[b] * P1_vals[c]) - (45360.0 * P7_vals[a] * P1_vals[c]) 
                             + (30240.0 * P5_vals[a] * P3_vals[c]) - (978075.0 * P3_vals[a] * P4_vals[b] * P1_vals[c]) 
                             - (916650.0 * P3_vals[a] * P2_vals[b] * P3_vals[c]) + (61425.0 * P3_vals[a] * P5_vals[c]) 
                             + (141750.0 * P1_vals[a] * P6_vals[b] * P1_vals[c]) + (269325.0 * P1_vals[a] * P4_vals[b] * P3_vals[c]) 
                             + (113400.0 * P1_vals[a] * P2_vals[b] * P5_vals[c]) - (14175.0 * P1_vals[a] * P7_vals[c]);
            add_to_atomic(value, base_key + 24 + pair, temp * gp6_term, hash_table, local_tid);
        }


        // group 7: 3 terms for pairs (x,y), (x,z), (y,z)
        constexpr int pairs3[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        #pragma unroll
        for (int pair = 0; pair < 3; ++pair) {
            const int a = pairs3[pair][0];
            const int b = pairs3[pair][1];
            const int c = 3 - a - b;
            const T gp7_term = (1119825.0 * P4_vals[a] * P4_vals[b]) - (438480.0 * P6_vals[a] * P2_vals[b]) 
                             - (141750.0 * P4_vals[a] * P2_vals[b] * P2_vals[c]) + (15120.0 * P8_vals[a]) 
                             + (15120.0 * P6_vals[a] * P2_vals[c]) - (14175.0 * P4_vals[a] * P4_vals[c]) 
                             - (438480.0 * P2_vals[a] * P6_vals[b]) - (141750.0 * P2_vals[a] * P4_vals[b] * P2_vals[c]) 
                             + (283500.0 * P2_vals[a] * P2_vals[b] * P4_vals[c]) - (13230.0 * P2_vals[a] * P6_vals[c]) 
                             + (15120.0 * P8_vals[b]) + (15120.0 * P6_vals[b] * P2_vals[c]) - (14175.0 * P4_vals[b] * P4_vals[c]) 
                             - (13230.0 * P2_vals[b] * P6_vals[c]) + (945.0 * P8_vals[c]);
            add_to_atomic(value, base_key + 30 + pair, temp * gp7_term, hash_table, local_tid);
        }


        // group 8: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp8_term = (1190700.0 * P4_vals[a] * P3_vals[b] * P1_vals[c]) - (226800.0 * P6_vals[a] * P1_vals[b] * P1_vals[c]) 
                             - (56700.0 * P4_vals[a] * P1_vals[b] * P3_vals[c]) - (586845.0 * P2_vals[a] * P5_vals[b] * P1_vals[c]) 
                             - (425250.0 * P2_vals[a] * P3_vals[b] * P3_vals[c]) + (161595.0 * P2_vals[a] * P1_vals[b] * P5_vals[c]) 
                             + (22680.0 * P7_vals[b] * P1_vals[c]) + (36855.0 * P5_vals[b] * P3_vals[c]) 
                             + (5670.0 * P3_vals[b] * P5_vals[c]) - (8505.0 * P1_vals[b] * P7_vals[c]);
            add_to_atomic(value, base_key + 33 + pair, temp * gp8_term, hash_table, local_tid);
        }


        // group 9: 3 terms for x, y, z
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp9_term = (1200150.0 * P4_vals[dim] * P2_vals[dim1] * P2_vals[dim2]) - (70560.0 * P6_vals[dim] * P2_vals[dim1]) 
                             - (23625.0 * P4_vals[dim] * P4_vals[dim1]) - (70560.0 * P6_vals[dim] * P2_vals[dim2]) 
                             - (23625.0 * P4_vals[dim] * P4_vals[dim2]) + (5040.0 * P8_vals[dim]) 
                             - (600075.0 * P2_vals[dim] * P4_vals[dim1] * P2_vals[dim2]) - (600075.0 * P2_vals[dim] * P2_vals[dim1] * P4_vals[dim2]) 
                             + (49455.0 * P2_vals[dim] * P6_vals[dim1]) + (49455.0 * P2_vals[dim] * P6_vals[dim2]) 
                             + (21105.0 * P6_vals[dim1] * P2_vals[dim2]) + (47250.0 * P4_vals[dim1] * P4_vals[dim2]) 
                             + (21105.0 * P2_vals[dim1] * P6_vals[dim2]) - (2520.0 * P8_vals[dim1]) - (2520.0 * P8_vals[dim2]);
            add_to_atomic(value, base_key + 39 + dim, temp * gp9_term, hash_table, local_tid);
        }


        // group 10: 3 terms for pairs (x,y), (x,z), (y,z)
        #pragma unroll
        for (int pair = 0; pair < 3; ++pair) {
            const int a = pairs3[pair][0];
            const int b = pairs3[pair][1];
            const int c = 3 - a - b;
            const T gp10_term = (1341900.0 * P3_vals[a] * P3_vals[b] * P2_vals[c]) - (67095.0 * P5_vals[a] * P3_vals[b]) 
                              - (67095.0 * P3_vals[a] * P5_vals[b]) - (274995.0 * P5_vals[a] * P1_vals[b] * P2_vals[c]) 
                              - (212625.0 * P3_vals[a] * P1_vals[b] * P4_vals[c]) + (22680.0 * P7_vals[a] * P1_vals[b]) 
                              - (274995.0 * P1_vals[a] * P5_vals[b] * P2_vals[c]) - (212625.0 * P1_vals[a] * P3_vals[b] * P4_vals[c]) 
                              + (22680.0 * P1_vals[a] * P7_vals[b]) + (85050.0 * P1_vals[a] * P1_vals[b] * P6_vals[c]);
            add_to_atomic(value, base_key + 42 + pair, temp * gp10_term, hash_table, local_tid);
        }

    }

    template <typename T>
    GPU_HOST_DEVICE void solid_mcsh_9(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, T* __restrict__ value, 
        const int base_key, const HashTable<T>* hash_table = nullptr, const int local_tid = 0) 
    {
        const array3d_t<T> lambda_dr = dr * lambda;
        const array3d_t<T> lambda_dr2 = lambda_dr * lambda_dr;
        const array3d_t<T> lambda_dr3 = lambda_dr2 * lambda_dr;
        const array3d_t<T> lambda_dr4 = lambda_dr3 * lambda_dr;
        const array3d_t<T> lambda_dr5 = lambda_dr4 * lambda_dr;
        const array3d_t<T> lambda_dr6 = lambda_dr5 * lambda_dr;
        const array3d_t<T> lambda_dr7 = lambda_dr6 * lambda_dr;
        const array3d_t<T> lambda_dr8 = lambda_dr7 * lambda_dr;
        const array3d_t<T> lambda_dr9 = lambda_dr8 * lambda_dr;
        const T inv_gamma = 1.0 / gamma;
        const T inv_gamma_2 = inv_gamma * inv_gamma;
        const T inv_gamma_3 = inv_gamma_2 * inv_gamma;
        const T inv_gamma_4 = inv_gamma_3 * inv_gamma;

        const array3d_t<T> P1_vals = lambda_dr;
        array3d_t<T> P2_vals;
        array3d_t<T> P3_vals;
        array3d_t<T> P4_vals;
        array3d_t<T> P5_vals;
        array3d_t<T> P6_vals;
        array3d_t<T> P7_vals;
        array3d_t<T> P8_vals;
        array3d_t<T> P9_vals;

        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            P2_vals[dim] = P2(lambda_dr2[dim], inv_gamma);
            P3_vals[dim] = P3(lambda_dr[dim], lambda_dr3[dim], inv_gamma);
            P4_vals[dim] = P4(lambda_dr2[dim], lambda_dr4[dim], inv_gamma, inv_gamma_2);
            P5_vals[dim] = P5(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], inv_gamma, inv_gamma_2);
            P6_vals[dim] = P6(lambda_dr2[dim], lambda_dr4[dim], lambda_dr6[dim], inv_gamma, inv_gamma_2, inv_gamma_3);
            P7_vals[dim] = P7(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], lambda_dr7[dim], inv_gamma, inv_gamma_2, inv_gamma_3);
            P8_vals[dim] = P8(lambda_dr2[dim], lambda_dr4[dim], lambda_dr6[dim], lambda_dr8[dim], inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
            P9_vals[dim] = P9(lambda_dr[dim], lambda_dr3[dim], lambda_dr5[dim], lambda_dr7[dim], lambda_dr9[dim], inv_gamma, inv_gamma_2, inv_gamma_3, inv_gamma_4);
        }

        // group 1: 3 terms for x, y, z
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp1_term = (362880.0 * P9_vals[dim]) - (6531840.0 * P7_vals[dim] * P2_vals[dim1]) 
                             - (6531840.0 * P7_vals[dim] * P2_vals[dim2]) + (17146080.0 * P5_vals[dim] * P4_vals[dim1]) 
                             + (34292160.0 * P5_vals[dim] * P2_vals[dim1] * P2_vals[dim2]) + (17146080.0 * P5_vals[dim] * P4_vals[dim2]) 
                             - (9525600.0 * P3_vals[dim] * P6_vals[dim1]) - (28576800.0 * P3_vals[dim] * P4_vals[dim1] * P2_vals[dim2]) 
                             - (28576800.0 * P3_vals[dim] * P2_vals[dim1] * P4_vals[dim2]) - (9525600.0 * P3_vals[dim] * P6_vals[dim2]) 
                             + (893025.0 * P1_vals[dim] * P8_vals[dim1]) + (3572100.0 * P1_vals[dim] * P6_vals[dim1] * P2_vals[dim2]) 
                             + (5358150.0 * P1_vals[dim] * P4_vals[dim1] * P4_vals[dim2]) + (3572100.0 * P1_vals[dim] * P2_vals[dim1] * P6_vals[dim2]) 
                             + (893025.0 * P1_vals[dim] * P8_vals[dim2]);
            add_to_atomic(value, base_key + dim, temp * gp1_term, hash_table, local_tid);
        }

        // group 2: 6 terms for pairs
        constexpr int pairs[6][2] = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 2}, {2, 1}};
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp2_term = (1814400.0 * P8_vals[a] * P1_vals[b]) - (12700800.0 * P6_vals[a] * P3_vals[b]) 
                             - (12700800.0 * P6_vals[a] * P1_vals[b] * P2_vals[c]) + (15876000.0 * P4_vals[a] * P5_vals[b]) 
                             + (31752000.0 * P4_vals[a] * P3_vals[b] * P2_vals[c]) + (15876000.0 * P4_vals[a] * P1_vals[b] * P4_vals[c]) 
                             - (3969000.0 * P2_vals[a] * P7_vals[b]) - (11907000.0 * P2_vals[a] * P5_vals[b] * P2_vals[c]) 
                             - (11907000.0 * P2_vals[a] * P3_vals[b] * P4_vals[c]) - (3969000.0 * P2_vals[a] * P1_vals[b] * P6_vals[c]) 
                             + (99225.0 * P9_vals[b]) + (396900.0 * P7_vals[b] * P2_vals[c]) + (595350.0 * P5_vals[b] * P4_vals[c]) 
                             + (396900.0 * P3_vals[b] * P6_vals[c]) + (99225.0 * P1_vals[b] * P8_vals[c]);
            add_to_atomic(value, base_key + 3 + pair, temp * gp2_term, hash_table, local_tid);
        }


        // group 3: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp3_term = (5760720.0 * P7_vals[a] * P2_vals[b]) - (181440.0 * P9_vals[a]) 
                             + (771120.0 * P7_vals[a] * P2_vals[c]) - (17304840.0 * P5_vals[a] * P4_vals[b]) 
                             - (17146080.0 * P5_vals[a] * P2_vals[b] * P2_vals[c]) + (158760.0 * P5_vals[a] * P4_vals[c]) 
                             + (10220175.0 * P3_vals[a] * P6_vals[b]) + (19745775.0 * P3_vals[a] * P4_vals[b] * P2_vals[c]) 
                             + (8831025.0 * P3_vals[a] * P2_vals[b] * P4_vals[c]) - (694575.0 * P3_vals[a] * P6_vals[c]) 
                             - (992250.0 * P1_vals[a] * P8_vals[b]) - (2877525.0 * P1_vals[a] * P6_vals[b] * P2_vals[c]) 
                             - (2679075.0 * P1_vals[a] * P4_vals[b] * P4_vals[c]) - (694575.0 * P1_vals[a] * P2_vals[b] * P6_vals[c]) 
                             + (99225.0 * P1_vals[a] * P8_vals[c]);
            add_to_atomic(value, base_key + 9 + pair, temp * gp3_term, hash_table, local_tid);
        }


        // group 4: 3 terms for x, y, z
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp4_term = (4989600.0 * P7_vals[dim] * P1_vals[dim1] * P1_vals[dim2]) 
                             - (17463600.0 * P5_vals[dim] * P3_vals[dim1] * P1_vals[dim2]) 
                             - (17463600.0 * P5_vals[dim] * P1_vals[dim1] * P3_vals[dim2]) 
                             + (10914750.0 * P3_vals[dim] * P5_vals[dim1] * P1_vals[dim2]) 
                             + (21829500.0 * P3_vals[dim] * P3_vals[dim1] * P3_vals[dim2]) 
                             + (10914750.0 * P3_vals[dim] * P1_vals[dim1] * P5_vals[dim2]) 
                             - (1091475.0 * P1_vals[dim] * P7_vals[dim1] * P1_vals[dim2]) 
                             - (3274425.0 * P1_vals[dim] * P5_vals[dim1] * P3_vals[dim2]) 
                             - (3274425.0 * P1_vals[dim] * P3_vals[dim1] * P5_vals[dim2]) 
                             - (1091475.0 * P1_vals[dim] * P1_vals[dim1] * P7_vals[dim2]);
            add_to_atomic(value, base_key + 15 + dim, temp * gp4_term, hash_table, local_tid);
        }


        // group 5: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp5_term = (12020400.0 * P6_vals[a] * P3_vals[b]) - (1360800.0 * P8_vals[a] * P1_vals[b]) 
                             + (2041200.0 * P6_vals[a] * P1_vals[b] * P2_vals[c]) - (16584750.0 * P4_vals[a] * P5_vals[b]) 
                             - (14458500.0 * P4_vals[a] * P3_vals[b] * P2_vals[c]) + (2126250.0 * P4_vals[a] * P1_vals[b] * P4_vals[c]) 
                             + (4380075.0 * P2_vals[a] * P7_vals[b]) + (7526925.0 * P2_vals[a] * P5_vals[b] * P2_vals[c]) 
                             + (1913625.0 * P2_vals[a] * P3_vals[b] * P4_vals[c]) - (1233225.0 * P2_vals[a] * P1_vals[b] * P6_vals[c]) 
                             - (113400.0 * P9_vals[b]) - (297675.0 * P7_vals[b] * P2_vals[c]) - (212625.0 * P5_vals[b] * P4_vals[c]) 
                             + (14175.0 * P3_vals[b] * P6_vals[c]) + (42525.0 * P1_vals[b] * P8_vals[c]);
            add_to_atomic(value, base_key + 18 + pair, temp * gp5_term, hash_table, local_tid);
        }


        // group 6: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp6_term = (10659600.0 * P6_vals[a] * P2_vals[b] * P1_vals[c]) - (453600.0 * P8_vals[a] * P1_vals[c]) 
                             + (680400.0 * P6_vals[a] * P3_vals[c]) - (18002250.0 * P4_vals[a] * P4_vals[b] * P1_vals[c]) 
                             - (17293500.0 * P4_vals[a] * P2_vals[b] * P3_vals[c]) + (708750.0 * P4_vals[a] * P5_vals[c]) 
                             + (5202225.0 * P2_vals[a] * P6_vals[b] * P1_vals[c]) + (9993375.0 * P2_vals[a] * P4_vals[b] * P3_vals[c]) 
                             + (4380075.0 * P2_vals[a] * P2_vals[b] * P5_vals[c]) - (411075.0 * P2_vals[a] * P7_vals[c]) 
                             - (141750.0 * P8_vals[b] * P1_vals[c]) - (411075.0 * P6_vals[b] * P3_vals[c]) 
                             - (382725.0 * P4_vals[b] * P5_vals[c]) - (99225.0 * P2_vals[b] * P7_vals[c]) + (14175.0 * P9_vals[c]);
            add_to_atomic(value, base_key + 24 + pair, temp * gp6_term, hash_table, local_tid);
        }


        // group 7: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp7_term = (19486005.0 * P5_vals[a] * P4_vals[b]) - (3412640.0 * P7_vals[a] * P2_vals[b]) 
                             + (5292210.0 * P5_vals[a] * P2_vals[b] * P2_vals[c]) + (518980.0 * P9_vals[a]) 
                             + (1576960.0 * P7_vals[a] * P2_vals[c]) + (2022405.0 * P5_vals[a] * P4_vals[c]) 
                             - (9524900.0 * P3_vals[a] * P6_vals[b]) - (1443750.0 * P3_vals[a] * P4_vals[b] * P2_vals[c]) 
                             + (9471000.0 * P3_vals[a] * P2_vals[b] * P4_vals[c]) + (1389850.0 * P3_vals[a] * P6_vals[c]) 
                             + (1516900.0 * P1_vals[a] * P8_vals[b]) + (2949100.0 * P1_vals[a] * P6_vals[b] * P2_vals[c]) 
                             + (1772925.0 * P1_vals[a] * P4_vals[b] * P4_vals[c]) + (766150.0 * P1_vals[a] * P2_vals[b] * P6_vals[c]) 
                             + (425425.0 * P1_vals[a] * P8_vals[c]);
            add_to_atomic(value, base_key + 30 + pair, temp * gp7_term, hash_table, local_tid);
        }


        // group 8: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp8_term = (16839900.0 * P5_vals[a] * P3_vals[b] * P1_vals[c]) - (2494800.0 * P7_vals[a] * P1_vals[b] * P1_vals[c]) 
                             + (623700.0 * P5_vals[a] * P1_vals[b] * P3_vals[c]) - (13565475.0 * P3_vals[a] * P5_vals[b] * P1_vals[c]) 
                             - (10914750.0 * P3_vals[a] * P3_vals[b] * P3_vals[c]) + (2650725.0 * P3_vals[a] * P1_vals[b] * P5_vals[c]) 
                             + (1559250.0 * P1_vals[a] * P7_vals[b] * P1_vals[c]) + (2650725.0 * P1_vals[a] * P5_vals[b] * P3_vals[c]) 
                             + (623700.0 * P1_vals[a] * P3_vals[b] * P5_vals[c]) - (467775.0 * P1_vals[a] * P1_vals[b] * P7_vals[c]);
            add_to_atomic(value, base_key + 36 + pair, temp * gp8_term, hash_table, local_tid);
        }


        // group 9: 3 terms for x, y, z
        #pragma unroll
        for (int dim = 0; dim < 3; ++dim) {
            const int dim1 = (dim + 1) % 3;
            const int dim2 = (dim + 2) % 3;
            const T gp9_term = (16448670.0 * P5_vals[dim] * P2_vals[dim1] * P2_vals[dim2]) - (816480.0 * P7_vals[dim] * P2_vals[dim1]) 
                             + (116235.0 * P5_vals[dim] * P4_vals[dim1]) - (816480.0 * P7_vals[dim] * P2_vals[dim2]) 
                             + (116235.0 * P5_vals[dim] * P4_vals[dim2]) + (45360.0 * P9_vals[dim]) 
                             - (13707225.0 * P3_vals[dim] * P4_vals[dim1] * P2_vals[dim2]) - (13707225.0 * P3_vals[dim] * P2_vals[dim1] * P4_vals[dim2]) 
                             + (836325.0 * P3_vals[dim] * P6_vals[dim1]) + (836325.0 * P3_vals[dim] * P6_vals[dim2]) 
                             + (1460025.0 * P1_vals[dim] * P6_vals[dim1] * P2_vals[dim2]) + (3203550.0 * P1_vals[dim] * P4_vals[dim1] * P4_vals[dim2]) 
                             + (1460025.0 * P1_vals[dim] * P2_vals[dim1] * P6_vals[dim2]) - (141750.0 * P1_vals[dim] * P8_vals[dim1]) 
                             - (141750.0 * P1_vals[dim] * P8_vals[dim2]);
            add_to_atomic(value, base_key + 42 + dim, temp * gp9_term, hash_table, local_tid);
        }


        // group 10: 3 terms for pairs (x,y), (x,z), (y,z)
        constexpr int pairs3[3][2] = {{0, 1}, {0, 2}, {1, 2}};
        #pragma unroll
        for (int pair = 0; pair < 3; ++pair) {
            const int a = pairs3[pair][0];
            const int b = pairs3[pair][1];
            const int c = 3 - a - b;
            const T gp10_term = (19604025.0 * P4_vals[a] * P4_vals[b] * P1_vals[c]) - (7200900.0 * P6_vals[a] * P2_vals[b] * P1_vals[c]) 
                              - (3203550.0 * P4_vals[a] * P2_vals[b] * P3_vals[c]) + (226800.0 * P8_vals[a] * P1_vals[c]) 
                              + (283500.0 * P6_vals[a] * P3_vals[c]) - (104895.0 * P4_vals[a] * P5_vals[c]) 
                              - (7200900.0 * P2_vals[a] * P6_vals[b] * P1_vals[c]) - (3203550.0 * P2_vals[a] * P4_vals[b] * P3_vals[c]) 
                              + (3844260.0 * P2_vals[a] * P2_vals[b] * P5_vals[c]) - (153090.0 * P2_vals[a] * P7_vals[c]) 
                              + (226800.0 * P8_vals[b] * P1_vals[c]) + (283500.0 * P6_vals[b] * P3_vals[c]) 
                              - (104895.0 * P4_vals[b] * P5_vals[c]) - (153090.0 * P2_vals[b] * P7_vals[c]) + (8505.0 * P9_vals[c]);
            add_to_atomic(value, base_key + 45 + pair, temp * gp10_term, hash_table, local_tid);
        }


        // group 11: 6 terms for pairs
        #pragma unroll
        for (int pair = 0; pair < 6; ++pair) {
            const int a = pairs[pair][0];
            const int b = pairs[pair][1];
            const int c = 3 - a - b;
            const T gp11_term = (20497050.0 * P4_vals[a] * P3_vals[b] * P2_vals[c]) - (963900.0 * P6_vals[a] * P3_vals[b]) 
                              - (603855.0 * P4_vals[a] * P5_vals[b]) - (3458700.0 * P6_vals[a] * P1_vals[b] * P2_vals[c]) 
                              - (1601775.0 * P4_vals[a] * P1_vals[b] * P4_vals[c]) + (226800.0 * P8_vals[a] * P1_vals[b]) 
                              - (8224335.0 * P2_vals[a] * P5_vals[b] * P2_vals[c]) - (6789825.0 * P2_vals[a] * P3_vals[b] * P4_vals[c]) 
                              + (564165.0 * P2_vals[a] * P7_vals[b]) + (1998675.0 * P2_vals[a] * P1_vals[b] * P6_vals[c]) 
                              + (252315.0 * P7_vals[b] * P2_vals[c]) + (487620.0 * P5_vals[b] * P4_vals[c]) 
                              + (127575.0 * P3_vals[b] * P6_vals[c]) - (22680.0 * P9_vals[b]) - (85050.0 * P1_vals[b] * P8_vals[c]);
            add_to_atomic(value, base_key + 48 + pair, temp * gp11_term, hash_table, local_tid);
        }

        // group 12
        const T gp12_term = (21829320.0 * P3_vals[0] * P3_vals[1] * P3_vals[2]) - (3274515.0 * P5_vals[0] * P3_vals[1] * P1_vals[2]) 
                          - (3274605.0 * P3_vals[0] * P5_vals[1] * P1_vals[2]) - (3274425.0 * P5_vals[0] * P1_vals[1] * P3_vals[2]) 
                          - (3274425.0 * P3_vals[0] * P1_vals[1] * P5_vals[2]) + (935550.0 * P7_vals[0] * P1_vals[1] * P1_vals[2]) 
                          - (3274605.0 * P1_vals[0] * P5_vals[1] * P3_vals[2]) - (3274515.0 * P1_vals[0] * P3_vals[1] * P5_vals[2]) 
                          + (935460.0 * P1_vals[0] * P7_vals[1] * P1_vals[2]) + (935550.0 * P1_vals[0] * P1_vals[1] * P7_vals[2]);
        add_to_atomic(value, base_key + 54, temp * gp12_term, hash_table, local_tid);
    }
}}