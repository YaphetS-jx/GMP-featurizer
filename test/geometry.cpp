#include <gtest/gtest.h>
#include "geometry.hpp"
#include <iostream>
#include <iomanip>


using namespace gmp::geometry;
using namespace gmp::math;

TEST(geometry, lattice_t_constructors) {
    // Default constructor
    lattice_t lat1;
    EXPECT_EQ(lat1[0], array3d_flt64(0, 0, 0));
    EXPECT_EQ(lat1[1], array3d_flt64(0, 0, 0));
    EXPECT_EQ(lat1[2], array3d_flt64(0, 0, 0));

    // Constructor with vectors and scaling
    array3d_flt64 v1(1, 0.5, 0.3);
    array3d_flt64 v2(0.2, 1.2, 0.4);
    array3d_flt64 v3(0.1, 0.3, 1.1);
    lattice_t lat2(v1, v2, v3, 2.0, 3.0, 4.0);
    EXPECT_EQ(lat2[0], array3d_flt64(2, 1, 0.6));
    EXPECT_EQ(lat2[1], array3d_flt64(0.6, 3.6, 1.2));
    EXPECT_EQ(lat2[2], array3d_flt64(0.4, 1.2, 4.4));

    // Constructor with matrix
    matrix3d_flt64 mat(v1, v2, v3);
    lattice_t lat3(mat);
    EXPECT_EQ(lat3[0], v1);
    EXPECT_EQ(lat3[1], v2);
    EXPECT_EQ(lat3[2], v3);

    // Copy constructor
    lattice_t lat4(lat2);
    EXPECT_EQ(lat4[0], lat2[0]);
    EXPECT_EQ(lat4[1], lat2[1]);
    EXPECT_EQ(lat4[2], lat2[2]);

    // Move constructor
    lattice_t lat5(std::move(lat3));
    EXPECT_EQ(lat5[0], v1);
    EXPECT_EQ(lat5[1], v2);
    EXPECT_EQ(lat5[2], v3);
}

TEST(geometry, lattice_t_assignment) {
    lattice_t lat1;
    array3d_flt64 v1(1, 0.5, 0.3);
    array3d_flt64 v2(0.2, 1.2, 0.4);
    array3d_flt64 v3(0.1, 0.3, 1.1);
    lattice_t lat2(v1, v2, v3);

    // Copy assignment
    lat1 = lat2;
    EXPECT_EQ(lat1[0], lat2[0]);
    EXPECT_EQ(lat1[1], lat2[1]);
    EXPECT_EQ(lat1[2], lat2[2]);

    // Move assignment
    lattice_t lat3;
    lat3 = std::move(lat2);
    EXPECT_EQ(lat3[0], v1);
    EXPECT_EQ(lat3[1], v2);
    EXPECT_EQ(lat3[2], v3);
}

TEST(geometry, lattice_t_access) {
    array3d_flt64 v1(1, 0.5, 0.3);
    array3d_flt64 v2(0.2, 1.2, 0.4);
    array3d_flt64 v3(0.1, 0.3, 1.1);
    lattice_t lat(v1, v2, v3);

    // Const access
    const lattice_t& clat = lat;
    EXPECT_EQ(clat[0], v1);
    EXPECT_EQ(clat[1], v2);
    EXPECT_EQ(clat[2], v3);

    // Non-const access
    lat[0] = array3d_flt64(2, 1, 0.6);
    EXPECT_EQ(lat[0], array3d_flt64(2, 1, 0.6));
}

TEST(geometry, lattice_t_volume) {
    // Create a triclinic lattice
    array3d_flt64 v1(1, 0.5, 0.3);
    array3d_flt64 v2(0.2, 1.2, 0.4);
    array3d_flt64 v3(0.1, 0.3, 1.1);
    lattice_t lat(v1, v2, v3);

    // Calculate expected volume using triple product
    double expected_volume = std::abs(v1.dot(v2.cross(v3)));
    EXPECT_DOUBLE_EQ(lat.get_volume(), expected_volume);

    // Test scaled lattice
    lattice_t lat2(v1, v2, v3, 2.0, 3.0, 4.0);
    EXPECT_DOUBLE_EQ(lat2.get_volume(), expected_volume * 2.0 * 3.0 * 4.0);
}

TEST(geometry, lattice_t_normalize) {
    // Create a triclinic lattice
    array3d_flt64 v1(2, 1, 0.6);
    array3d_flt64 v2(0.6, 3.6, 1.2);
    array3d_flt64 v3(0.4, 1.2, 4.4);
    lattice_t lat(v1, v2, v3);

    // Get normalized vectors
    matrix3d_flt64 norm = lat.normalize();

    // Check that vectors are unit length
    EXPECT_DOUBLE_EQ(norm[0].norm(), 1.0);
    EXPECT_DOUBLE_EQ(norm[1].norm(), 1.0);
    EXPECT_DOUBLE_EQ(norm[2].norm(), 1.0);

    // Check directions are preserved (vectors should be parallel to original)
    array3d_flt64 cross1 = norm[0].cross(v1);
    array3d_flt64 cross2 = norm[1].cross(v2);
    array3d_flt64 cross3 = norm[2].cross(v3);
    EXPECT_NEAR(cross1.norm(), 0.0, 1e-10);
    EXPECT_NEAR(cross2.norm(), 0.0, 1e-10);
    EXPECT_NEAR(cross3.norm(), 0.0, 1e-10);
}

TEST(geometry, lattice_t_distance) {
    // Create a triclinic lattice
    array3d_flt64 v1(8.2143131137332475, 0.0000000000000000, 0.0000000000000000);
    array3d_flt64 v2(0.0000000000000000, 7.3692440000000001, 0.0000000000000000);
    array3d_flt64 v3(-5.0606965770721848, 0.0000000000000000, 7.3320024020115309);
    lattice_t lat(v1, v2, v3);

    // Update metric tensor
    lat.update_metric();
    
    point_flt64 p1 = {1.2, 3.4, 5.6};
    point_flt64 p2 = {6.5, 2.3, 4.3};
    array3d_flt64 cell_shift(0, 0, 0);
    array3d_flt64 difference;
    
    double distance2 = lat.calculate_distance_squared(p1, p2, cell_shift, difference);    
    EXPECT_DOUBLE_EQ(distance2, 2668.0511005315929651);
}

TEST(geometry, lattice_t_cell_info_to_lattice) {
    array3d_flt64 cell_lengths(8.2143131137332475, 7.3692440000000001, 8.9089230027143564);
    array3d_flt64 cell_angles(90.0000000000000000, 124.6142240624760120, 90.0000000000000000);
    lattice_t lat = cell_info_to_lattice(cell_lengths, cell_angles);
    EXPECT_EQ(lat[0], array3d_flt64(8.2143131137332475, 0.0000000000000000, 0.0000000000000000));
    EXPECT_EQ(lat[1], array3d_flt64(0.0000000000000000, 7.3692440000000001, 0.0000000000000000));
    EXPECT_EQ(lat[2], array3d_flt64(-5.0606965770721848, 0.0000000000000000, 7.3320024020115309));

    // test volume
    EXPECT_DOUBLE_EQ(lat.get_volume(), 443.8301369664612821);
    
    // Test interplanar spacing
    array3d_flt64 spacing = lat.get_interplanar_spacing();
    EXPECT_GT(spacing[0], 0.0);
    EXPECT_GT(spacing[1], 0.0);
    EXPECT_GT(spacing[2], 0.0);
    
    // Test cell lengths
    array3d_flt64 lengths = lat.get_cell_lengths();
    EXPECT_DOUBLE_EQ(lengths[0], cell_lengths[0]);
    EXPECT_DOUBLE_EQ(lengths[1], cell_lengths[1]);
    EXPECT_DOUBLE_EQ(lengths[2], cell_lengths[2]);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 