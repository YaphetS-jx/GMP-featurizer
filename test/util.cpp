#include <gtest/gtest.h>
#include "util.hpp"
#include "error.hpp"

TEST(UtilTest, ParseLineBasicTypes) {
    int x;
    double y;
    std::string z;
    
    gmp::util::parse_line("42, 3.14, hello", x, y, z);
    EXPECT_EQ(x, 42);
    EXPECT_DOUBLE_EQ(y, 3.14);
    EXPECT_EQ(z, "hello");
}

TEST(UtilTest, ParseLineWithWhitespace) {
    int a;
    double b;
    std::string c;
    
    gmp::util::parse_line("  123  ,   4.56   ,   world  ", a, b, c);
    EXPECT_EQ(a, 123);
    EXPECT_DOUBLE_EQ(b, 4.56);
    EXPECT_EQ(c, "world");
}

// Test for non-streamable type
struct NonStreamable {
    int value;
};

TEST(UtilTest, ParseLineNonStreamable) {
    int x;
    NonStreamable ns;
    std::string z;
    
    gmp::util::parse_line("42, 99, test", x, ns, z);
    // First value should be parsed
    EXPECT_EQ(x, 42);
    // Last value should not be parsed due to error return
    EXPECT_NE(z, "test");
    // Check if error was set
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::unstreamable_string);
}

TEST(UtilTest, ParseLineEmpty) {
    int x = 0;
    double y = 0.0;
    
    gmp::util::parse_line("", x, y);
    EXPECT_EQ(x, 0);
    EXPECT_DOUBLE_EQ(y, 0.0);
}

TEST(UtilTest, ParseLinePatternSingleType) {
    // Test single type (vector of int)
    auto ints = gmp::util::parse_line_pattern<int>("1, 2, 3, 4, 5");
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::success);
    EXPECT_EQ(ints.size(), 5);
    EXPECT_EQ(ints[0], 1);
    EXPECT_EQ(ints[4], 5);

    // Test single type with whitespace
    auto nums = gmp::util::parse_line_pattern<double>("  1.1  ,  2.2  ,  3.3  ");
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::success);
    EXPECT_EQ(nums.size(), 3);
    EXPECT_DOUBLE_EQ(nums[0], 1.1);
    EXPECT_DOUBLE_EQ(nums[2], 3.3);
}

TEST(UtilTest, ParseLinePatternMultiType) {

    // Test complete set with two types
    auto pairs = gmp::util::parse_line_pattern<int, double>("1, 1.1, 2, 2.2");
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::success);
    EXPECT_EQ(pairs.size(), 2);
    EXPECT_EQ(std::get<0>(pairs[0]), 1);
    EXPECT_DOUBLE_EQ(std::get<1>(pairs[0]), 1.1);
    EXPECT_EQ(std::get<0>(pairs[1]), 2);
    EXPECT_DOUBLE_EQ(std::get<1>(pairs[1]), 2.2);

    // Test three types with complete set
    auto triples = gmp::util::parse_line_pattern<int, double, std::string>(
        "1, 1.1, one, 2, 2.2, two"
    );
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::success);
    EXPECT_EQ(triples.size(), 2);
    EXPECT_EQ(std::get<0>(triples[0]), 1);
    EXPECT_DOUBLE_EQ(std::get<1>(triples[0]), 1.1);
    EXPECT_EQ(std::get<2>(triples[0]), "one");

    
    // Test two types with incomplete set
    pairs = gmp::util::parse_line_pattern<int, double>("1, 1.1, 2, 2.2, 3");
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::incomplete_data_set);
    EXPECT_TRUE(pairs.empty());

    // Test three types with incomplete set
    triples = gmp::util::parse_line_pattern<int, double, std::string>(
        "1, 1.1, one, 2, 2.2"  // Missing last string
    );
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::incomplete_data_set);
    EXPECT_TRUE(triples.empty());
}

TEST(UtilTest, ParseLinePatternEmpty) {
    // Test empty input
    auto empty_ints = gmp::util::parse_line_pattern<int>("");
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::success);
    EXPECT_TRUE(empty_ints.empty());

    auto empty_pairs = gmp::util::parse_line_pattern<int, double>("");
    EXPECT_EQ(gmp::get_last_error(), gmp::error_t::success);
    EXPECT_TRUE(empty_pairs.empty());
} 