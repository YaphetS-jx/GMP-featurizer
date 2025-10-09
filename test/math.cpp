#include <gtest/gtest.h>
#include "math.hpp"
#include "resources.hpp"
#include "mcsh.hpp"

using namespace gmp::math;
using namespace gmp::resources;

template <typename T>
bool compare_array3d(const array3d_t<T>& a, const array3d_t<T>& b) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
}

template <typename T>
bool compare_matrix3d(const matrix3d_t<T>& a, const matrix3d_t<T>& b) {
    return compare_array3d(a[0], b[0]) 
        && compare_array3d(a[1], b[1]) 
        && compare_array3d(a[2], b[2]);
}

template <typename T>
bool compare_sym_matrix3d(const sym_matrix3d_t<T>& a, const sym_matrix3d_t<T>& b) {
    return compare_array3d(a.diag_, b.diag_) 
        && compare_array3d(a.off_diag_, b.off_diag_);
}

TEST(math, array3d_t) {
    // Test constructors
    array3d_t<double> zero;  // default constructor
    array3d_t<double> a = make_array3d(1.0, 2.0, 3.0);  // value constructor
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
    array3d_t<double> b = make_array3d(2.0, 3.0, 4.0);
    array3d_t<double> sum = a + b;
    EXPECT_TRUE(compare_array3d(sum, make_array3d(3.0, 5.0, 7.0)));

    array3d_t<double> diff = a - b;
    EXPECT_TRUE(compare_array3d(diff, make_array3d(-1.0, -1.0, -1.0)));

    array3d_t<double> scaled = a * 2.0;
    EXPECT_TRUE(compare_array3d(scaled, make_array3d(2.0, 4.0, 6.0)));

    array3d_t<double> scaled2 = 2.0 * a;
    EXPECT_TRUE(compare_array3d(scaled2, make_array3d(2.0, 4.0, 6.0)));

    array3d_t<double> divided = a / 2.0;
    EXPECT_TRUE(compare_array3d(divided, make_array3d(0.5, 1.0, 1.5)));

    // Test comparison operators
    array3d_t<double> a2 = make_array3d(1.0, 2.0, 3.0);
    EXPECT_TRUE(compare_array3d(a, a2));

    // Test dot product
    double dot = a.dot(b);
    EXPECT_DOUBLE_EQ(dot, 20.0);  // 1*2 + 2*3 + 3*4 = 20

    // Test cross product
    array3d_t<double> cross = a.cross(b);
    EXPECT_TRUE(compare_array3d(cross, make_array3d(-1.0, 2.0, -1.0)));

    // Test norm
    array3d_t<double> c = make_array3d(3.0, 4.0, 0.0);
    EXPECT_DOUBLE_EQ(c.norm(), 5.0);
}

TEST(math, matrix3d_t) {
    // Test constructors
    matrix3d_t<double> zero;  // default constructor
    matrix3d_t<double> a = make_matrix3d(make_array3d(1.0, 2.0, 3.0), 
                                        make_array3d(4.0, 5.0, 6.0), 
                                        make_array3d(7.0, 8.0, 9.0));
    matrix3d_t<double> a_copy(a);  // copy constructor
    matrix3d_t<double> a_move(std::move(a_copy));  // move constructor
    
    // Test assignment operators
    matrix3d_t<double> assign_copy;
    assign_copy = a;  // copy assignment
    matrix3d_t<double> assign_move;
    assign_move = std::move(assign_copy);  // move assignment
    
    // Test matrix access
    EXPECT_TRUE(compare_array3d(a[0], make_array3d(1.0, 2.0, 3.0)));
    EXPECT_TRUE(compare_array3d(a[1], make_array3d(4.0, 5.0, 6.0)));
    EXPECT_TRUE(compare_array3d(a[2], make_array3d(7.0, 8.0, 9.0)));
    
    // Test comparison operators
    matrix3d_t<double> a2 = make_matrix3d(make_array3d(1.0, 2.0, 3.0),
                                         make_array3d(4.0, 5.0, 6.0),
                                         make_array3d(7.0, 8.0, 9.0));
    EXPECT_TRUE(compare_matrix3d(a, a2));
    
    // Test matrix multiplication
    matrix3d_t<double> b = make_matrix3d(make_array3d(2.0, 0.0, 1.0),
                                        make_array3d(0.0, 2.0, 0.0),
                                        make_array3d(1.0, 0.0, 2.0));
    matrix3d_t<double> ab_expected = make_matrix3d(make_array3d(5.0, 4.0, 7.0),
                                                 make_array3d(14.0, 10.0, 16.0),
                                                 make_array3d(23.0, 16.0, 25.0));
    EXPECT_TRUE(compare_matrix3d(a * b, ab_expected));

    // Test matrix-vector multiplication
    array3d_t<double> v = make_array3d(1.0, 2.0, 3.0);
    array3d_t<double> av_expected = make_array3d(14.0, 32.0, 50.0);
    EXPECT_TRUE(compare_array3d(a * v, av_expected));
    
    // Test transpose
    matrix3d_t<double> a_trans_expected = make_matrix3d(make_array3d(1.0, 4.0, 7.0),
                                                      make_array3d(2.0, 5.0, 8.0),
                                                      make_array3d(3.0, 6.0, 9.0));
    EXPECT_TRUE(compare_matrix3d(a.transpose(), a_trans_expected));
    
    // Test determinant
    matrix3d_t<double> det_test = make_matrix3d(make_array3d(2.0, -1.0, 0.0),
                                               make_array3d(-1.0, 2.0, -1.0),
                                               make_array3d(0.0, -1.0, 2.0));
    EXPECT_DOUBLE_EQ(det_test.det(), 4.0);
    
    // Test inverse
    matrix3d_t<double> inv_expected = make_matrix3d(make_array3d(0.75, 0.5, 0.25),
                                                  make_array3d(0.5, 1.0, 0.5),
                                                  make_array3d(0.25, 0.5, 0.75));
    auto inv_result = det_test.inverse();
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            EXPECT_NEAR(inv_result[i][j], inv_expected[i][j], 1e-10);
        }
    }
    
}

TEST(math, sym_matrix3d_t) {
    // Test constructors and accessors
    sym_matrix3d_t<double> s;
    s[0] = 1.0; s[1] = 2.0; s[2] = 3.0; // diagonal elements
    s[3] = 4.0; s[4] = 5.0; s[5] = 6.0; // off-diagonal elements
    EXPECT_DOUBLE_EQ(s[0], 1.0); // diagonal elements
    EXPECT_DOUBLE_EQ(s[1], 2.0);
    EXPECT_DOUBLE_EQ(s[2], 3.0);
    EXPECT_DOUBLE_EQ(s[3], 4.0); // off-diagonal elements
    EXPECT_DOUBLE_EQ(s[4], 5.0);
    EXPECT_DOUBLE_EQ(s[5], 6.0);

    // Test array3d constructor
    sym_matrix3d_t<double> s2;
    s2.diag_ = make_array3d(1.0, 2.0, 3.0);
    s2.off_diag_ = make_array3d(4.0, 5.0, 6.0);
    EXPECT_TRUE(compare_sym_matrix3d(s, s2));

    // Test copy constructor
    sym_matrix3d_t<double> s3(s);
    EXPECT_TRUE(compare_sym_matrix3d(s, s3));

    // Test assignment operator
    sym_matrix3d_t<double> s4;
    s4 = s;
    EXPECT_TRUE(compare_sym_matrix3d(s, s4));

    // Test comparison operators
    EXPECT_TRUE(compare_sym_matrix3d(s, s2));

    // Test matrix-vector multiplication
    array3d_t<double> v = make_array3d(1.0, 2.0, 3.0);
    array3d_t<double> sv_expected = make_array3d(
        1.0*1.0 + 4.0*2.0 + 5.0*3.0,  // first row
        4.0*1.0 + 2.0*2.0 + 6.0*3.0,  // second row
        5.0*1.0 + 6.0*2.0 + 3.0*3.0   // third row
    );
    EXPECT_TRUE(compare_array3d(s * v, sv_expected));
};

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 