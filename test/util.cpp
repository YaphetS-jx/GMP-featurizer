#include <gtest/gtest.h>
#include "util.hpp"
#include "error.hpp"

TEST(UtilTest, SortIndexes) {
    // Test with integers
    std::vector<int> int_data = {5, 2, 8, 1, 9};
    auto int_indices = gmp::util::sort_indexes<int, int, std::vector>(int_data);
    EXPECT_EQ(int_indices.size(), 5);
    EXPECT_EQ(int_indices[0], 3);  // index of 1
    EXPECT_EQ(int_indices[1], 1);  // index of 2
    EXPECT_EQ(int_indices[2], 0);  // index of 5
    EXPECT_EQ(int_indices[3], 2);  // index of 8
    EXPECT_EQ(int_indices[4], 4);  // index of 9

    // Test with floating point numbers
    std::vector<double> double_data = {3.14, 1.41, 2.71, 0.0};
    auto double_indices = gmp::util::sort_indexes<double, int, std::vector>(double_data);
    EXPECT_EQ(double_indices.size(), 4);
    EXPECT_EQ(double_indices[0], 3);  // index of 0.0
    EXPECT_EQ(double_indices[1], 1);  // index of 1.41
    EXPECT_EQ(double_indices[2], 2);  // index of 2.71
    EXPECT_EQ(double_indices[3], 0);  // index of 3.14

    // Test with equal values (should maintain stable order)
    std::vector<int> equal_data = {2, 2, 1, 2};
    auto equal_indices = gmp::util::sort_indexes<int, int, std::vector>(equal_data);
    EXPECT_EQ(equal_indices.size(), 4);
    EXPECT_EQ(equal_indices[0], 2);  // index of 1
    EXPECT_EQ(equal_indices[1], 0);  // index of first 2
    EXPECT_EQ(equal_indices[2], 1);  // index of second 2
    EXPECT_EQ(equal_indices[3], 3);  // index of third 2

    // Test with empty vector
    std::vector<int> empty_data;
    auto empty_indices = gmp::util::sort_indexes<int, int, std::vector>(empty_data);
    EXPECT_TRUE(empty_indices.empty());
} 