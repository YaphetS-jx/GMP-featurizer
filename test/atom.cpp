#include <gtest/gtest.h>
#include "atom.hpp"
#include "resources.hpp"

using namespace gmp;
using namespace gmp::atom;
using namespace gmp::geometry;
using namespace gmp::containers;

// Test atom_t class
TEST(atom_system, atom_constructors) {
    // Default constructor with values
    atom_t atom1(1.0, 2.0, 3.0);
    EXPECT_DOUBLE_EQ(atom1.x(), 1.0);
    EXPECT_DOUBLE_EQ(atom1.y(), 2.0);
    EXPECT_DOUBLE_EQ(atom1.z(), 3.0);
    EXPECT_DOUBLE_EQ(atom1.occ(), 1.0);
    EXPECT_EQ(atom1.id(), std::numeric_limits<uint8_t>::max());

    // Constructor with point and custom values
    point_flt64 pos{4.0, 5.0, 6.0};
    atom_t atom2(pos, 0.5, 2);
    EXPECT_DOUBLE_EQ(atom2.x(), 4.0);
    EXPECT_DOUBLE_EQ(atom2.y(), 5.0);
    EXPECT_DOUBLE_EQ(atom2.z(), 6.0);
    EXPECT_DOUBLE_EQ(atom2.occ(), 0.5);
    EXPECT_EQ(atom2.id(), 2);

    // Copy constructor
    atom_t atom3(atom1);
    EXPECT_DOUBLE_EQ(atom3.x(), 1.0);
    EXPECT_DOUBLE_EQ(atom3.y(), 2.0);
    EXPECT_DOUBLE_EQ(atom3.z(), 3.0);

    // Move constructor
    atom_t atom4(std::move(atom2));
    EXPECT_DOUBLE_EQ(atom4.x(), 4.0);
    EXPECT_DOUBLE_EQ(atom4.y(), 5.0);
    EXPECT_DOUBLE_EQ(atom4.z(), 6.0);
}

TEST(atom_system, atom_assignment) {
    atom_t atom1(1.0, 2.0, 3.0);
    atom_t atom2(4.0, 5.0, 6.0);

    // Copy assignment
    atom1 = atom2;
    EXPECT_DOUBLE_EQ(atom1.x(), 4.0);
    EXPECT_DOUBLE_EQ(atom1.y(), 5.0);
    EXPECT_DOUBLE_EQ(atom1.z(), 6.0);

    // Move assignment
    atom_t atom3(7.0, 8.0, 9.0);
    atom1 = std::move(atom3);
    EXPECT_DOUBLE_EQ(atom1.x(), 7.0);
    EXPECT_DOUBLE_EQ(atom1.y(), 8.0);
    EXPECT_DOUBLE_EQ(atom1.z(), 9.0);
}

// Test unit_cell_t class
TEST(atom, system_constructors) {
    auto& host_memory = gmp_resource::instance(128, 1<<20).get_host_memory();
    // Default constructor
    unit_cell_t system1;
    EXPECT_TRUE(system1.get_atoms().empty());
    EXPECT_FALSE(system1.get_lattice());
    EXPECT_TRUE(system1.get_periodicity()[0]);
    EXPECT_TRUE(system1.get_periodicity()[1]);
    EXPECT_TRUE(system1.get_periodicity()[2]);

    // Constructor with values
    vec<atom_t> atoms;
    atoms.push_back(atom_t(1.0, 2.0, 3.0));
    atoms.push_back(atom_t(4.0, 5.0, 6.0));

    auto lattice = make_gmp_unique<lattice_t>();
    array3d_bool periodic{false, true, false};

    unit_cell_t system2(std::move(atoms), std::move(lattice), periodic);
    EXPECT_EQ(system2.get_atoms().size(), 2);
    EXPECT_FALSE(system2.get_periodicity()[0]);
    EXPECT_TRUE(system2.get_periodicity()[1]);
    EXPECT_FALSE(system2.get_periodicity()[2]);
}

TEST(atom, system_mutators) {
    auto& host_memory = gmp_resource::instance(128, 1<<20).get_host_memory();
    unit_cell_t system;

    // Test set_atoms
    vec<atom_t> atoms;
    atoms.push_back(atom_t(1.0, 2.0, 3.0));
    system.set_atoms(std::move(atoms));
    EXPECT_EQ(system.get_atoms().size(), 1);
    EXPECT_DOUBLE_EQ(system[0].x(), 1.0);

    // Test set_lattice
    auto lattice = make_gmp_unique<lattice_t>();
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
    auto& host_memory = gmp_resource::instance(128, 1<<20).get_host_memory();
    
    // Create system with some atoms
    vec<atom_t> atoms;
    atoms.push_back(atom_t(1.0, 2.0, 3.0));
    atoms.push_back(atom_t(4.0, 5.0, 6.0));
    auto lattice = make_gmp_unique<lattice_t>();
    array3d_bool periodic{false, true, false};
    
    unit_cell_t system(std::move(atoms), std::move(lattice), periodic);

    // Test get_atoms
    const auto& system_atoms = system.get_atoms();
    EXPECT_EQ(system_atoms.size(), 2);

    // Test get_lattice
    EXPECT_TRUE(system.get_lattice());

    // Test get_periodicity
    const auto& system_periodic = system.get_periodicity();
    EXPECT_FALSE(system_periodic[0]);
    EXPECT_TRUE(system_periodic[1]);
    EXPECT_FALSE(system_periodic[2]);

    // Test operator[]
    EXPECT_DOUBLE_EQ(system[0].x(), 1.0);
    EXPECT_DOUBLE_EQ(system[1].x(), 4.0);

    // Test const operator[]
    const unit_cell_t& const_system = system;
    EXPECT_DOUBLE_EQ(const_system[0].x(), 1.0);
    EXPECT_DOUBLE_EQ(const_system[1].x(), 4.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 