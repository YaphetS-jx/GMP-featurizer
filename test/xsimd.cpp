#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>
#include "types.hpp"
#include "simd.hpp"

using namespace gmp::simd;

// Test basic vector addition
TEST(XSimdTest, VectorAddition) {
    const size_t size = 1000;
    vector_aligned<double> a(size, 1.0);
    vector_aligned<double> b(size, 2.0);
    vector_aligned<double> result(size);

    // Using xsimd for vectorized addition
    using batch_type = xsimd::batch<double>;
    const size_t batch_size = batch_type::size;
    
    for (size_t i = 0; i < size; i += batch_size) {
        auto batch_a = batch_type::load_aligned(&a[i]);
        auto batch_b = batch_type::load_aligned(&b[i]);
        auto batch_result = batch_a + batch_b;
        batch_result.store_aligned(&result[i]);
    }

    // Verify results
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(result[i], 3.0);
    }
}

// Test vector multiplication and addition (similar to MCSH operations)
TEST(XSimdTest, VectorMultiplyAdd) {
    const size_t size = 1000;
    vector_aligned<double> a(size);
    vector_aligned<double> b(size);
    vector_aligned<double> c(size);
    vector_aligned<double> result(size);

    // Initialize with some values
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<double>(i);
        b[i] = 2.0;
        c[i] = 1.0;
    }

    // Using xsimd for vectorized multiply-add
    using batch_type = xsimd::batch<double>;
    const size_t batch_size = batch_type::size;
    
    for (size_t i = 0; i < size; i += batch_size) {
        auto batch_a = batch_type::load_aligned(&a[i]);
        auto batch_b = batch_type::load_aligned(&b[i]);
        auto batch_c = batch_type::load_aligned(&c[i]);
        auto batch_result = batch_a * batch_b + batch_c;
        batch_result.store_aligned(&result[i]);
    }

    // Verify results
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(result[i], a[i] * b[i] + c[i]);
    }
}

// Test vectorized exponential and multiplication (similar to MCSH P1-P9)
TEST(XSimdTest, VectorExpMultiply) {
    const size_t size = 1000;
    vector_aligned<double> a(size);
    vector_aligned<double> b(size);
    vector_aligned<double> result(size);

    // Initialize with some values
    for (size_t i = 0; i < size; ++i) {
        a[i] = static_cast<double>(i) * 0.1;
        b[i] = 2.0;
    }

    // Using xsimd for vectorized exp and multiply
    using batch_type = xsimd::batch<double>;
    const size_t batch_size = batch_type::size;
    
    for (size_t i = 0; i < size; i += batch_size) {
        auto batch_a = batch_type::load_aligned(&a[i]);
        auto batch_b = batch_type::load_aligned(&b[i]);
        auto batch_result = xsimd::exp(batch_a) * batch_b;
        batch_result.store_aligned(&result[i]);
    }

    // Verify results
    for (size_t i = 0; i < size; ++i) {
        EXPECT_DOUBLE_EQ(result[i], std::exp(a[i]) * b[i]);
    }
}

// Test vectorized reduction (sum)
TEST(XSimdTest, VectorReduction) {
    const size_t size = 1000;
    vector_aligned<double> data(size);

    // Initialize with some values
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Using xsimd for vectorized reduction
    using batch_type = xsimd::batch<double>;
    const size_t batch_size = batch_type::size;
    
    batch_type sum_batch = batch_type(0.0);
    for (size_t i = 0; i < size; i += batch_size) {
        auto batch_data = batch_type::load_aligned(&data[i]);
        sum_batch += batch_data;
    }

    double sum = xsimd::reduce_add(sum_batch);

    // Verify result
    double expected_sum = std::accumulate(data.begin(), data.end(), 0.0);
    EXPECT_DOUBLE_EQ(sum, expected_sum);
}
