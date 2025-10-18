#include <gtest/gtest.h>
#include "atom.hpp"
#include "containers.hpp"

using namespace gmp;
using namespace gmp::atom;
using namespace gmp::geometry;
using namespace gmp::containers;

// Macro for floating point comparisons that works with both single and double precision
#ifdef GMP_USE_SINGLE_PRECISION
#define EXPECT_FLOAT_EQ_GMP EXPECT_FLOAT_EQ
#else
#define EXPECT_FLOAT_EQ_GMP EXPECT_DOUBLE_EQ
#endif

// Test atom_t class
TEST(atom_system, atom_assignment) {
    atom_t<gmp::gmp_float> atom1{{1.0, 2.0, 3.0}, 1.0, 0};
    atom_t<gmp::gmp_float> atom2{{4.0, 5.0, 6.0}, 1.0, 1};

    // Copy assignment
    atom1 = atom2;
    EXPECT_FLOAT_EQ_GMP(atom1.pos.x, 4.0);
    EXPECT_FLOAT_EQ_GMP(atom1.pos.y, 5.0);
    EXPECT_FLOAT_EQ_GMP(atom1.pos.z, 6.0);
}

TEST(atom, system_mutators) {
    unit_cell_t<gmp::gmp_float> system;

    // Test set_atoms
    std::vector<atom_t<gmp::gmp_float>> atoms;
    atoms.push_back({{1.0, 2.0, 3.0}, 1.0, 0});
    system.set_atoms(std::move(atoms));
    EXPECT_EQ(system.get_atoms().size(), 1);
    EXPECT_FLOAT_EQ_GMP(system[0].pos.x, 1.0);
    EXPECT_FLOAT_EQ_GMP(system[0].pos.y, 2.0);
    EXPECT_FLOAT_EQ_GMP(system[0].pos.z, 3.0);

    // Test set_lattice
    matrix3d_t<gmp::gmp_float> mat;
    mat[0] = array3d_t<gmp::gmp_float>{1.0, 0.0, 0.0};
    mat[1] = array3d_t<gmp::gmp_float>{0.0, 1.0, 0.0};
    mat[2] = array3d_t<gmp::gmp_float>{0.0, 0.0, 1.0};
    auto lattice = std::make_unique<lattice_t<gmp::gmp_float>>(mat);
    system.set_lattice(std::move(lattice));
    EXPECT_TRUE(system.get_lattice());

    // Test set_periodicity
    array3d_bool periodic{true, false, true};
    system.set_periodicity(periodic);
    EXPECT_TRUE(system.get_periodicity()[0]);
    EXPECT_FALSE(system.get_periodicity()[1]);
    EXPECT_TRUE(system.get_periodicity()[2]);
}

TEST(atom, system_accessors) {
    // Create system with some atoms
    std::vector<atom_t<gmp::gmp_float>> atoms;
    atoms.push_back({{1.0, 2.0, 3.0}, 1.0, 0});
    atoms.push_back({{4.0, 5.0, 6.0}, 1.0, 1});
    
    matrix3d_t<gmp::gmp_float> mat;
    mat[0] = array3d_t<gmp::gmp_float>{1.0, 0.0, 0.0};
    mat[1] = array3d_t<gmp::gmp_float>{0.0, 1.0, 0.0};
    mat[2] = array3d_t<gmp::gmp_float>{0.0, 0.0, 1.0};
    auto lattice = std::make_unique<lattice_t<gmp::gmp_float>>(mat);
    
    atom_type_map_t atom_type_map;
    atom_type_map["H"] = 0;
    atom_type_map["O"] = 1;
    
    unit_cell_t<gmp::gmp_float> system;
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
    EXPECT_FLOAT_EQ_GMP(system[0].pos.x, 1.0);
    EXPECT_FLOAT_EQ_GMP(system[1].pos.x, 4.0);

    // Test const operator[]
    const unit_cell_t<gmp::gmp_float>& const_system = system;
    EXPECT_FLOAT_EQ_GMP(const_system[0].pos.x, 1.0);
    EXPECT_FLOAT_EQ_GMP(const_system[1].pos.x, 4.0);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 