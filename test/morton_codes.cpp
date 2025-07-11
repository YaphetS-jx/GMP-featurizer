#include <gtest/gtest.h>
#include "morton_codes.hpp"
#include <cmath>

using namespace gmp::tree::morton_codes;

// Helper functions to handle template parameters
uint32_t fractional_to_binary_wrapper(double num, uint32_t num_bits) {
    return fractional_to_binary<double, uint32_t, uint32_t>(num, num_bits);
}

uint32_t coordinate_to_morton_code_wrapper(double num, uint32_t num_bits) {
    return coordinate_to_morton_code<double, uint32_t, uint32_t>(num, num_bits);
}

double binary_to_fractional_wrapper(uint32_t binary, uint32_t num_bits) {
    return binary_to_fractional<uint32_t, uint32_t, double>(binary, num_bits);
}

// Test print_bits function
TEST(MortonCodesTest, PrintBits) {
    // Test with uint32_t
    uint32_t num = 0b1010;
    testing::internal::CaptureStdout();
    print_bits(num, 4);
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_EQ(output, "1010\n");  // Note: bits are printed from least significant to most significant
}

// Test interleave_bits function
TEST(MortonCodesTest, InterleaveBits) {
    uint32_t x = 0b101;  // 5
    uint32_t y = 0b110;  // 6
    uint32_t z = 0b111;  // 7
    uint32_t num_bits = 3;

    print_bits(x, 3);
    print_bits(y, 3);
    print_bits(z, 3);
    
    uint32_t result = interleave_bits(x, y, z, num_bits);
    // Expected: 111 110 101 (binary) = 0x1F5
    EXPECT_EQ(result, 0x1F5);
}

// Test deinterleave_bits function
TEST(MortonCodesTest, BitDeinterleaveBits) {
    uint32_t morton_code = 0x1F5;  // 111 110 101
    uint32_t num_bits_per_dim = 3;
    uint32_t x, y, z;

    deinterleave_bits(morton_code, num_bits_per_dim, x, y, z);
    EXPECT_EQ(x, 0b101);  // 5
    EXPECT_EQ(y, 0b110);  // 6
    EXPECT_EQ(z, 0b111);  // 7
}

// Test fractional_to_binary function
TEST(MortonCodesTest, FractionalToBinary) {
    double num = 0.625;  // 0.101 in binary
    uint32_t num_bits = 3;
    
    uint32_t result = fractional_to_binary_wrapper(num, num_bits);
    EXPECT_EQ(result, 0b101);
}

// Test coordinate_to_morton_code function
TEST(MortonCodesTest, CoordinateToMortonCode) {
    double num = 0.625;  // 0.101 in binary
    uint32_t num_bits = 4;
    
    uint32_t result = coordinate_to_morton_code_wrapper(num, num_bits);
    EXPECT_EQ(result, 0b101);
}

// Test binary_to_fractional function
TEST(MortonCodesTest, BinaryToFractional) {
    uint32_t binary = 0b101;  // 0.625 in decimal
    uint32_t num_bits = 3;
    
    double result = binary_to_fractional_wrapper(binary, num_bits);
    EXPECT_DOUBLE_EQ(result, 0.625);
}

// Test morton_code_to_coordinate function
TEST(MortonCodesTest, MortonCodeToCoordinate) {
    uint32_t morton_code = 0b101;  // 0.625 in decimal
    uint32_t num_bits = 4;
    
    double result = morton_code_to_coordinate<double, uint32_t, uint32_t>(morton_code, num_bits);
    EXPECT_DOUBLE_EQ(result, 0.625);
}

// Test edge cases
TEST(MortonCodesTest, EdgeCases) {
    // Test with minimum values
    EXPECT_EQ(fractional_to_binary_wrapper(0.0, 3), 0);
    EXPECT_EQ(coordinate_to_morton_code_wrapper(0.0, 4), 0);
    
    // Test with values close to 1.0
    double close_to_one = 0.999999;
    uint32_t result = fractional_to_binary_wrapper(close_to_one, 10);
    EXPECT_EQ(result, 0b1111111111);
    
    // Test with maximum bits
    uint32_t max_bits = 32;
    double small_value = 1.0 / (1ULL << max_bits);
    result = fractional_to_binary_wrapper(small_value, max_bits);
    EXPECT_EQ(result, 1);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 