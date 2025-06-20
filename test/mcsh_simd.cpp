#include <gtest/gtest.h>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>
#include "simd.hpp"
#include "mcsh_simd.hpp"
#include "math.hpp"
#include "types.hpp"

using namespace gmp::simd;
using namespace gmp::containers;

// Helper function to create random parameters
template<typename T>
Params<T> create_random_params(size_t length, T min_val = -1.0, T max_val = 1.0) {
    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(11);
    std::uniform_real_distribution<T> dist(min_val, max_val);
    
    Params<T> params;
    params.num_elements = length;
    
    // Initialize all vectors
    params.dx.resize(length);
    params.dy.resize(length);
    params.dz.resize(length);
    params.r_sqr.resize(length);
    params.C1.resize(length);
    params.C2.resize(length);
    params.lambda.resize(length);
    params.gamma.resize(length);
    
    // Fill with random values
    for (size_t i = 0; i < length; ++i) {
        params.dx[i] = dist(gen);
        params.dy[i] = dist(gen);
        params.dz[i] = dist(gen);
        params.r_sqr[i] = std::abs(dist(gen)); // r_sqr should be positive
        params.C1[i] = dist(gen);
        params.C2[i] = dist(gen);
        params.lambda[i] = dist(gen);
        params.gamma[i] = dist(gen);
    }
    
    return params;
}


TEST(MCSHSIMDTest, test_order_0) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[0];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_0(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << duration.count() << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_0<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}


TEST(MCSHSIMDTest, test_order_1) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[1];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_1(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_1<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}



TEST(MCSHSIMDTest, test_order_2) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[2];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_2(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_2<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}


TEST(MCSHSIMDTest, test_order_3) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[3];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_3(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_3<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}

TEST(MCSHSIMDTest, test_order_4) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[4];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_4(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_4<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}

TEST(MCSHSIMDTest, test_order_5) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[5];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_5(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_5<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}

TEST(MCSHSIMDTest, test_order_6) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[6];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_6(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_6<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}

TEST(MCSHSIMDTest, test_order_7) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[7];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_7(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_7<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}

TEST(MCSHSIMDTest, test_order_8) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[8];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_8(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_8<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}

TEST(MCSHSIMDTest, test_order_9) {
    const size_t length = 1033;
    const auto params = create_random_params<double>(length);
    const auto num_values = gmp::simd::num_values_[9];

    auto start = std::chrono::high_resolution_clock::now();
    // Initialize output vector with the correct size for order 0
    vector<double> out_ref(num_values, 0.0);
    
    for (size_t i = 0; i < length; ++i) {
        gmp::math::array3d_flt64 dr = {params.dx[i], params.dy[i], params.dz[i]};
        double temp = params.C1[i] * std::exp(params.C2[i] * params.r_sqr[i]);
        gmp::math::calculate_solid_mcsh_9(
            dr, params.r_sqr[i], temp, params.lambda[i], params.gamma[i], out_ref);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto time1 = duration.count();
    std::cout << "Time1: " << time1 << " sec" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    vector_aligned<double> out_simd(num_values, 0.0);
    // test simd
    gmp::simd::mcsh_simd_launcher_9<double>()(params, out_simd);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    auto time2 = duration.count();
    std::cout << "Time2: " << time2 << " sec" << std::endl;
    std::cout << "Speedup: " << time1 / time2 << std::endl;

    for (size_t i = 0; i < num_values; ++i) {
        EXPECT_NEAR(out_ref[i], out_simd[i], 1e-8*std::abs(out_simd[i]));
    }
}