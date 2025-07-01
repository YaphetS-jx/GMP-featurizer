#include <gtest/gtest.h>
#include "atom.hpp"
#include "types.hpp"

using namespace gmp;
using namespace gmp::atom;
using namespace gmp::geometry;
using namespace gmp::containers;

// Test atom_t class
TEST(atom_system, atom_constructors) {
    // Default constructor with values
    atom_flt64 atom1(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(atom1.x(), 1.0);
    EXPECT_DOUBLE_EQ(atom1.y(), 2.0);
    EXPECT_DOUBLE_EQ(atom1.z(), 3.0);
    EXPECT_DOUBLE_EQ(atom1.occ(), 1.0);
    EXPECT_EQ(atom1.id(), std::numeric_limits<uint8_t>::max());

    // Constructor with point and custom values
    point_flt64 pos{4.0, 5.0, 6.0};
    atom_flt64 atom2(pos, 0.5, 2);
    EXPECT_DOUBLE_EQ(atom2.x(), 4.0);
    EXPECT_DOUBLE_EQ(atom2.y(), 5.0);
    EXPECT_DOUBLE_EQ(atom2.z(), 6.0);
    EXPECT_DOUBLE_EQ(atom2.occ(), 0.5);
    EXPECT_EQ(atom2.id(), 2);

    // Copy constructor
    atom_flt64 atom3(atom1);
    EXPECT_DOUBLE_EQ(atom3.x(), 1.0);
    EXPECT_DOUBLE_EQ(atom3.y(), 2.0);
    EXPECT_DOUBLE_EQ(atom3.z(), 3.0);
}

TEST(atom_system, atom_assignment) {
    atom_flt64 atom1(1.0, 2.0, 3.0);
    atom_flt64 atom2(4.0, 5.0, 6.0);

    // Copy assignment
    atom1 = atom2;
    EXPECT_DOUBLE_EQ(atom1.x(), 4.0);
    EXPECT_DOUBLE_EQ(atom1.y(), 5.0);
    EXPECT_DOUBLE_EQ(atom1.z(), 6.0);
}

TEST(atom, system_mutators) {
    unit_cell_flt64 system;

    // Test set_atoms
    vector<atom_flt64> atoms;
    atoms.push_back(atom_flt64(1.0, 2.0, 3.0));
    system.set_atoms(std::move(atoms));
    EXPECT_EQ(system.get_atoms().size(), 1);
    EXPECT_DOUBLE_EQ(system[0].x(), 1.0);

    // Test set_lattice
    matrix3d_flt64 mat;
    mat[0] = array3d_flt64{1.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 1.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 1.0};
    auto lattice = std::make_unique<lattice_flt64>(mat);
    system.set_lattice(std::move(lattice));
    EXPECT_TRUE(system.get_lattice());

    // Test set_periodicity
    array3d_bool periodic{false, true, false};
    system.set_periodicity(periodic);
    EXPECT_FALSE(system.get_periodicity()[0]);
    EXPECT_TRUE(system.get_periodicity()[1]);
    EXPECT_FALSE(system.get_periodicity()[2]);
}

TEST(atom, system_accessors) {
    // Create system with some atoms
    vector<atom_flt64> atoms;
    atoms.push_back(atom_flt64(1.0, 2.0, 3.0));
    atoms.push_back(atom_flt64(4.0, 5.0, 6.0));
    
    matrix3d_flt64 mat;
    mat[0] = array3d_flt64{1.0, 0.0, 0.0};
    mat[1] = array3d_flt64{0.0, 1.0, 0.0};
    mat[2] = array3d_flt64{0.0, 0.0, 1.0};
    auto lattice = std::make_unique<lattice_flt64>(mat);
    
    atom_type_map_t atom_type_map;
    atom_type_map["H"] = 0;
    atom_type_map["O"] = 1;
    
    unit_cell_flt64 system;
    system.set_atoms(std::move(atoms));
    system.set_lattice(std::move(lattice));
    system.set_atom_type_map(std::move(atom_type_map));
    system.set_periodicity(array3d_bool{true, false, true});

    // Test get_atoms
    const auto& system_atoms = system.get_atoms();
    EXPECT_EQ(system_atoms.size(), 2);

    // Test get_lattice
    EXPECT_TRUE(system.get_lattice());

    // Test get_periodicity
    const auto& system_periodic = system.get_periodicity();
    EXPECT_TRUE(system_periodic[0]);
    EXPECT_FALSE(system_periodic[1]);
    EXPECT_TRUE(system_periodic[2]);

    // Test operator[]
    EXPECT_DOUBLE_EQ(system[0].x(), 1.0);
    EXPECT_DOUBLE_EQ(system[1].x(), 4.0);

    // Test const operator[]
    const unit_cell_flt64& const_system = system;
    EXPECT_DOUBLE_EQ(const_system[0].x(), 1.0);
    EXPECT_DOUBLE_EQ(const_system[1].x(), 4.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 