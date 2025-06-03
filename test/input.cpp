#include <gtest/gtest.h>
#include "input.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>

// Define PROJECT_ROOT if not already defined by CMake
#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."
#endif

using namespace gmp::input;
using namespace gmp::math;
using namespace gmp::atom;
using namespace gmp::geometry;
// Removed using namespace gmp to avoid ambiguity with error_t

auto& pool = gmp_resource::instance(64, 1<<20).get_host_memory().get_pool();

// Helper function to construct paths relative to project root
std::string get_project_path(const std::string& relative_path) {
    return std::filesystem::path(PROJECT_ROOT) / relative_path;
}

// Reset error state before each test
class InputTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset error state before each test
        gmp::update_error(gmp::error_t::success);
    }
};

TEST_F(InputTest, file_path_t) {
    file_path_t file_paths;
    
    // Test setters
    file_paths.set_atom_file("test/test_files/test.cif");
    file_paths.set_psp_file("test/test_files/test.gpsp");
    file_paths.set_output_file("output.dat");
    
    // Test getters
    EXPECT_EQ(file_paths.get_atom_file(), "test/test_files/test.cif");
    EXPECT_EQ(file_paths.get_psp_file(), "test/test_files/test.gpsp");
    EXPECT_EQ(file_paths.get_output_file(), "output.dat");
}

TEST_F(InputTest, descriptor_config_t) {    
    descriptor_config_t config;
    
    // Test default values
    EXPECT_EQ(config.get_feature_list().size(), 0);
    EXPECT_EQ(config.get_cutoff_method(), cutoff_method_t::cutoff_feature_gaussian);
    EXPECT_EQ(config.get_scaling_mode(), scaling_mode_t::radial);
    EXPECT_DOUBLE_EQ(config.get_cutoff(), 0.0);
    EXPECT_DOUBLE_EQ(config.get_overlap_threshold(), 1e-11);
    EXPECT_FALSE(config.get_square());
    
    // Test setters
    std::vector<int> orders = {0, 1};
    std::vector<double> sigmas = {0.1, 0.2};
    std::vector<std::tuple<int, double>> feature_list;
    
    config.set_feature_list(orders, sigmas, feature_list);
    // Each order is combined with each sigma: 2 orders * 2 sigmas = 4 features
    EXPECT_EQ(config.get_feature_list().size(), 4);
    
    config.set_cutoff_method(cutoff_method_t::cutoff_sigma);
    EXPECT_EQ(config.get_cutoff_method(), cutoff_method_t::cutoff_sigma);
    
    config.set_scaling_mode(scaling_mode_t::both);
    EXPECT_EQ(config.get_scaling_mode(), scaling_mode_t::both);
    
    config.set_cutoff(5.0);
    EXPECT_DOUBLE_EQ(config.get_cutoff(), 5.0);
    
    config.set_overlap_threshold(1e-10);
    EXPECT_DOUBLE_EQ(config.get_overlap_threshold(), 1e-10);
    
    config.set_square(true);
    EXPECT_TRUE(config.get_square());
}

TEST_F(InputTest, read_atom_file) {
    // Test reading a CIF file
    std::unique_ptr<lattice_t> lattice;
    vec<atom_t> atoms;
    atom_type_map_t atom_type_map;
    
    // Use path relative to project root
    std::string cif_path = get_project_path("test/test_files/test.cif");
    read_atom_file(cif_path, lattice, atoms, atom_type_map);
    
    // Check if error occurred
    EXPECT_EQ(gmp::gmp_error, gmp::error_t::success);
    
    // Check lattice parameters
    EXPECT_DOUBLE_EQ(lattice->get_cell_lengths()[0], 8.214313113733247);
    EXPECT_DOUBLE_EQ(lattice->get_cell_lengths()[1], 7.369244);
    EXPECT_DOUBLE_EQ(lattice->get_cell_lengths()[2], 8.908923002714356);
    
    // Check atoms
    EXPECT_EQ(atoms.size(), 16); // 16 atoms in the CIF file
    
    // Check atom types
    EXPECT_EQ(atom_type_map.size(), 3); // K, Se, O
    EXPECT_EQ(atom_type_map["K"], 0);
    EXPECT_EQ(atom_type_map["Se"], 1);
    EXPECT_EQ(atom_type_map["O"], 2);
    
    // Check first atom position using accessors
    EXPECT_DOUBLE_EQ(atoms[0].x(), 0.140197);
    EXPECT_DOUBLE_EQ(atoms[3].y(), 0.421795);
    EXPECT_DOUBLE_EQ(atoms[8].z(), 0.010882);
    EXPECT_DOUBLE_EQ(atoms[0].occ(), 1.0);
    EXPECT_EQ(atoms[1].id(), atom_type_map["K"]);
}

TEST_F(InputTest, read_psp_file) {
    // First set up atom type map
    atom_type_map_t atom_type_map;
    atom_type_map["K"] = 0;
    atom_type_map["Se"] = 1;
    atom_type_map["O"] = 2;
    
    // Test reading a PSP file
    vec<gaussian_t> gaussian_table;
    vec<int> offset;

    // Use path relative to project root
    std::string psp_path = get_project_path("test/test_files/test.gpsp");
    read_psp_file(psp_path, atom_type_map, gaussian_table, offset);
    
    // Check if error occurred
    EXPECT_EQ(gmp::gmp_error, gmp::error_t::success);
    
    // Check gaussian_table size
    EXPECT_EQ(gaussian_table.size(), 17); // K, Se, O
    
    // Check if data was loaded for each atom type
    EXPECT_EQ(offset.size(), 4);
}

TEST_F(InputTest, input_t_parse_arguments) {
    // Create input object with path relative to project root
    input_t input(get_project_path("test/test_files/input_test.json"));
    
    // Check if error occurred
    EXPECT_EQ(gmp::gmp_error, gmp::error_t::success);
    
    // Check parsed values
    EXPECT_EQ(input.files->get_atom_file(), "test/test_files/test.cif");
    EXPECT_EQ(input.files->get_psp_file(), "test/test_files/test.gpsp");
    EXPECT_EQ(input.files->get_output_file(), "output.dat");
    
    // Each order is combined with each sigma: 2 orders * 2 sigmas = 4 features
    EXPECT_EQ(input.descriptor_config->get_feature_list().size(), 4);
    EXPECT_EQ(input.descriptor_config->get_cutoff_method(), cutoff_method_t::cutoff_sigma);
    EXPECT_DOUBLE_EQ(input.descriptor_config->get_cutoff(), 5.0);
    EXPECT_DOUBLE_EQ(input.descriptor_config->get_overlap_threshold(), 1e-10);
    EXPECT_EQ(input.descriptor_config->get_scaling_mode(), scaling_mode_t::radial);
    EXPECT_TRUE(input.descriptor_config->get_square());
}

TEST_F(InputTest, to_string_functions) {
    // Test cutoff_method_t to_string
    EXPECT_EQ(gmp_to_string(cutoff_method_t::custom_cutoff), "custom_cutoff");
    EXPECT_EQ(gmp_to_string(cutoff_method_t::cutoff_sigma), "cutoff_sigma");
    EXPECT_EQ(gmp_to_string(cutoff_method_t::cutoff_sigma_elemental), "cutoff_sigma_elemental");
    EXPECT_EQ(gmp_to_string(cutoff_method_t::cutoff_feature_elemental), "cutoff_feature_elemental");
    EXPECT_EQ(gmp_to_string(cutoff_method_t::cutoff_feature_gaussian), "cutoff_feature_gaussian");
    
    // Test scaling_mode_t to_string
    EXPECT_EQ(gmp_to_string(scaling_mode_t::radial), "radial");
    EXPECT_EQ(gmp_to_string(scaling_mode_t::both), "both");
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 