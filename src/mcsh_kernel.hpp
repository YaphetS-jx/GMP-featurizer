#pragma once
#include "types.hpp"
#include "math.hpp"

namespace gmp { namespace mcsh {

    using gmp::math::array3d_t;
    using gmp::containers::vector;

    // Optimized polynomial calculations using constexpr
    template <typename T>
    constexpr T P1(const T lambda_x0);
    
    template <typename T>
    constexpr T P2(const T lambda_x0_2, const T inv_gamma);
    
    template <typename T>
    constexpr T P3(const T lambda_x0, const T lambda_x0_3, const T inv_gamma);
    
    template <typename T>
    constexpr T P4(const T lambda_x0_2, const T lambda_x0_4, const T inv_gamma, const T inv_gamma_2);
    
    template <typename T>
    constexpr T P5(const T lambda_x0, const T lambda_x0_3, const T lambda_x0_5, const T inv_gamma, const T inv_gamma_2);
    
    template <typename T>
    constexpr T P6(const T lambda_x0_2, const T lambda_x0_4, const T lambda_x0_6, 
        const T inv_gamma, const T inv_gamma_2, const T inv_gamma_3);
    
    template <typename T>
    constexpr T P7(const T lambda_x0, const T lambda_x0_3, const T lambda_x0_5, const T lambda_x0_7, 
        const T inv_gamma, const T inv_gamma_2, const T inv_gamma_3);
    
    template <typename T>
    constexpr T P8(const T lambda_x0_2, const T lambda_x0_4, const T lambda_x0_6, const T lambda_x0_8, 
        const T inv_gamma, const T inv_gamma_2, const T inv_gamma_3, const T inv_gamma_4);
    
    template <typename T>
    constexpr T P9(const T lambda_x0, const T lambda_x0_3, const T lambda_x0_5, const T lambda_x0_7, const T lambda_x0_9, 
        const T inv_gamma,  const T inv_gamma_2, const T inv_gamma_3, const T inv_gamma_4);

    template <typename T>
    void solid_mcsh_0(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_1(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_2(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_3(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_4(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_5(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_6(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_7(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_8(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);

    template <typename T>
    void solid_mcsh_9(const array3d_t<T>& dr, const T r_sqr, const T temp, const T lambda, const T gamma, vector<T>& value);    
}}