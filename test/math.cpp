#include <gtest/gtest.h>
#include "math.hpp"
#include "resources.hpp"
using namespace gmp::math;
using namespace gmp::resources;

TEST(math, array3d_t) {
    // Test constructors
    array3d_t<double> zero;  // default constructor
    array3d_t<double> a(1.0, 2.0, 3.0);  // value constructor
    array3d_t<double> a_copy(a);  // copy constructor
    array3d_t<double> a_move(std::move(a_copy));  // move constructor

    // Test assignment operators
    array3d_t<double> assign_copy;
    assign_copy = a;  // copy assignment
    array3d_t<double> assign_move;
    assign_move = std::move(assign_copy);  // move assignment

    // Test array access
    EXPECT_DOUBLE_EQ(a[0], 1.0);
    EXPECT_DOUBLE_EQ(a[1], 2.0);
    EXPECT_DOUBLE_EQ(a[2], 3.0);

    // Test arithmetic operations
    array3d_t<double> b(2.0, 3.0, 4.0);
    array3d_t<double> sum = a + b;
    EXPECT_EQ(sum, array3d_t<double>(3.0, 5.0, 7.0));

    array3d_t<double> diff = a - b;
    EXPECT_EQ(diff, array3d_t<double>(-1.0, -1.0, -1.0));

    array3d_t<double> scaled = a * 2.0;
    EXPECT_EQ(scaled, array3d_t<double>(2.0, 4.0, 6.0));

    array3d_t<double> scaled2 = 2.0 * a;
    EXPECT_EQ(scaled2, array3d_t<double>(2.0, 4.0, 6.0));

    array3d_t<double> divided = a / 2.0;
    EXPECT_EQ(divided, array3d_t<double>(0.5, 1.0, 1.5));

    // Test comparison operators
    array3d_t<double> a2(1.0, 2.0, 3.0);
    EXPECT_TRUE(a == a2);
    EXPECT_FALSE(a != a2);

    // Test dot product
    double dot = a.dot(b);
    EXPECT_DOUBLE_EQ(dot, 20.0);  // 1*2 + 2*3 + 3*4 = 20

    // Test cross product
    array3d_t<double> cross = a.cross(b);
    EXPECT_EQ(cross, array3d_t<double>(-1.0, 2.0, -1.0));

    // Test norm
    array3d_t<double> c(3.0, 4.0, 0.0);
    EXPECT_DOUBLE_EQ(c.norm(), 5.0);
}

TEST(math, matrix3d_t) {
    // Test constructors
    matrix3d_t<double> zero;  // default constructor
    matrix3d_t<double> a(array3d_t<double>(1.0, 2.0, 3.0), 
                        array3d_t<double>(4.0, 5.0, 6.0), 
                        array3d_t<double>(7.0, 8.0, 9.0));
    matrix3d_t<double> a_copy(a);  // copy constructor
    matrix3d_t<double> a_move(std::move(a_copy));  // move constructor
    
    // Test assignment operators
    matrix3d_t<double> assign_copy;
    assign_copy = a;  // copy assignment
    matrix3d_t<double> assign_move;
    assign_move = std::move(assign_copy);  // move assignment
    
    // Test matrix access
    EXPECT_EQ(a[0], array3d_t<double>(1.0, 2.0, 3.0));
    EXPECT_EQ(a[1], array3d_t<double>(4.0, 5.0, 6.0));
    EXPECT_EQ(a[2], array3d_t<double>(7.0, 8.0, 9.0));
    
    // Test comparison operators
    matrix3d_t<double> a2(array3d_t<double>(1.0, 2.0, 3.0),
                         array3d_t<double>(4.0, 5.0, 6.0),
                         array3d_t<double>(7.0, 8.0, 9.0));
    EXPECT_TRUE(a == a2);
    EXPECT_FALSE(a != a2);
    
    // Test matrix multiplication
    matrix3d_t<double> b(array3d_t<double>(2.0, 0.0, 1.0),
                        array3d_t<double>(0.0, 2.0, 0.0),
                        array3d_t<double>(1.0, 0.0, 2.0));
    matrix3d_t<double> ab_expected(array3d_t<double>(5.0, 4.0, 7.0),
                                 array3d_t<double>(14.0, 10.0, 16.0),
                                 array3d_t<double>(23.0, 16.0, 25.0));
    EXPECT_EQ(a * b, ab_expected);
    
    // Test matrix-vector multiplication
    array3d_t<double> v(1.0, 2.0, 3.0);
    array3d_t<double> av_expected(14.0, 32.0, 50.0);
    EXPECT_EQ(a * v, av_expected);
    
    // Test transpose
    matrix3d_t<double> a_trans_expected(array3d_t<double>(1.0, 4.0, 7.0),
                                      array3d_t<double>(2.0, 5.0, 8.0),
                                      array3d_t<double>(3.0, 6.0, 9.0));
    EXPECT_EQ(a.transpose(), a_trans_expected);
    
    // Test determinant
    matrix3d_t<double> det_test(array3d_t<double>(2.0, -1.0, 0.0),
                               array3d_t<double>(-1.0, 2.0, -1.0),
                               array3d_t<double>(0.0, -1.0, 2.0));
    EXPECT_DOUBLE_EQ(det_test.det(), 4.0);
    
    // Test inverse
    matrix3d_t<double> inv_expected(array3d_t<double>(0.75, 0.5, 0.25),
                                  array3d_t<double>(0.5, 1.0, 0.5),
                                  array3d_t<double>(0.25, 0.5, 0.75));
    auto inv_result = det_test.inverse();
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            EXPECT_NEAR(inv_result[i][j], inv_expected[i][j], 1e-10);
        }
    }
    
}

TEST(math, sym_matrix3d_t) {
    // Test constructors and accessors
    sym_matrix3d_t<double> s(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    EXPECT_DOUBLE_EQ(s[0], 1.0); // diagonal elements
    EXPECT_DOUBLE_EQ(s[1], 2.0);
    EXPECT_DOUBLE_EQ(s[2], 3.0);
    EXPECT_DOUBLE_EQ(s[3], 4.0); // off-diagonal elements
    EXPECT_DOUBLE_EQ(s[4], 5.0);
    EXPECT_DOUBLE_EQ(s[5], 6.0);

    // Test array3d constructor
    sym_matrix3d_t<double> s2(array3d_t<double>(1.0, 2.0, 3.0), 
                             array3d_t<double>(4.0, 5.0, 6.0));
    EXPECT_EQ(s, s2);

    // Test copy constructor
    sym_matrix3d_t<double> s3(s);
    EXPECT_EQ(s, s3);

    // Test assignment operator
    sym_matrix3d_t<double> s4;
    s4 = s;
    EXPECT_EQ(s, s4);

    // Test comparison operators
    EXPECT_TRUE(s == s2);
    EXPECT_FALSE(s != s2);

    // Test matrix-vector multiplication
    array3d_t<double> v(1.0, 2.0, 3.0);
    array3d_t<double> sv_expected(
        1.0*1.0 + 4.0*2.0 + 5.0*3.0,  // first row
        4.0*1.0 + 2.0*2.0 + 6.0*3.0,  // second row
        5.0*1.0 + 6.0*2.0 + 3.0*3.0   // third row
    );
    EXPECT_EQ(s * v, sv_expected);
};

// Very basic test for mcsh_function_registry_t
TEST(math, mcsh_function_registry_t) {
    // Use the test-friendly instance instead of the regular one
    auto& registry = mcsh_function_registry_t::get_instance();
    
    // Now we can safely test the values without memory allocation issues
    EXPECT_EQ(registry.get_num_values(0), 1);
    EXPECT_EQ(registry.get_num_values(1), 3);
    EXPECT_EQ(registry.get_num_values(2), 6);
    EXPECT_EQ(registry.get_num_values(3), 10);
    EXPECT_EQ(registry.get_num_values(4), 15);
    EXPECT_EQ(registry.get_num_values(5), 21);
    EXPECT_EQ(registry.get_num_values(6), 28);
    EXPECT_EQ(registry.get_num_values(7), 36);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 