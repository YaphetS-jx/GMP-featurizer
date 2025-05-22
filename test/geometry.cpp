#include <gtest/gtest.h>
#include "geometry.hpp"
#include "math.hpp"
#include <iostream>
#include <iomanip>

using namespace gmp;
using namespace gmp::geometry;
using namespace gmp::math;

// Test lattice_t class
TEST(geometry, lattice_t_constructors) {
    // Constructor with matrix
    matrix3d_flt64 mat;
    mat[0] = array3d_flt64{1.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 1.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 1.0};
    lattice_t lat2(mat);
    EXPECT_DOUBLE_EQ(lat2.get_volume(), 1.0);

    // Copy constructor
    lattice_t lat3(lat2);
    EXPECT_DOUBLE_EQ(lat3.get_volume(), 1.0);
}

TEST(geometry, lattice_t_assignment) {
    matrix3d_flt64 mat;
    mat[0] = array3d_flt64{1.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 1.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 1.0};
    lattice_t lat1(mat);
    lattice_t lat2(mat);  // Create with same matrix instead of default
    lat2 = lat1;
    EXPECT_DOUBLE_EQ(lat2.get_volume(), 1.0);
}

TEST(geometry, lattice_t_access) {
    matrix3d_flt64 mat;
    mat[0] = array3d_flt64{1.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 1.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 1.0};
    lattice_t lat(mat);
    EXPECT_DOUBLE_EQ(lat[0][0], 1.0);
    EXPECT_DOUBLE_EQ(lat[1][1], 1.0);
    EXPECT_DOUBLE_EQ(lat[2][2], 1.0);
}

TEST(geometry, lattice_t_volume) {
    matrix3d_flt64 mat;
    mat[0] = array3d_flt64{1.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 1.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 1.0};
    lattice_t lat(mat);
    EXPECT_DOUBLE_EQ(lat.get_volume(), 1.0);

    mat[0] = array3d_flt64{2.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 3.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 4.0};
    lattice_t lat2(mat);
    EXPECT_DOUBLE_EQ(lat2.get_volume(), 24.0);
}

TEST(geometry, lattice_t_normalize) {
    matrix3d_flt64 mat;
    mat[0] = array3d_flt64{2.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 2.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 2.0};
    lattice_t lat(mat);
    auto norm = lat.normalize();
    EXPECT_DOUBLE_EQ(norm[0].norm(), 1.0);
    EXPECT_DOUBLE_EQ(norm[1].norm(), 1.0);
    EXPECT_DOUBLE_EQ(norm[2].norm(), 1.0);
}

TEST(geometry, lattice_t_distance) {
    matrix3d_flt64 mat;
    mat[0] = array3d_flt64{1.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 1.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 1.0};
    lattice_t lat(mat);
    point_flt64 p1{0.0, 0.0, 0.0};
    point_flt64 p2{1.0, 1.0, 1.0};
    array3d_flt64 cell_shift{0.0, 0.0, 0.0};
    array3d_flt64 difference;
    EXPECT_DOUBLE_EQ(lat.calculate_distance_squared(p1, p2, cell_shift, difference), 3.0);
}

TEST(geometry, lattice_t_cell_info_to_lattice) {
    array3d_flt64 cell_lengths{1.0, 1.0, 1.0};
    array3d_flt64 cell_angles{90.0, 90.0, 90.0};
    auto lat = cell_info_to_lattice(cell_lengths, cell_angles);
    EXPECT_DOUBLE_EQ(lat->get_volume(), 1.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 