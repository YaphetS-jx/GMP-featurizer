#include <gtest/gtest.h>
#include "input.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <boost/json.hpp>
#include <fstream>
#include <sstream>

// Define PROJECT_ROOT if not already defined by CMake
#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."
#endif

using namespace gmp::input;
using namespace gmp::math;
using namespace gmp::atom;
using namespace gmp::geometry;

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
    descriptor_config_t<double> config;
    
    // Test default values
    EXPECT_EQ(config.get_feature_list().size(), 0);
    EXPECT_EQ(config.get_scaling_mode(), scaling_mode_t::radial);
    EXPECT_DOUBLE_EQ(config.get_overlap_threshold(), 1e-11);
    EXPECT_FALSE(config.get_square());
    
    // Test setters
    std::vector<int> orders = {0, 1};
    std::vector<double> sigmas = {0.1, 0.2};
    std::vector<std::tuple<int, double>> feature_list;
    
    config.set_feature_list(orders, sigmas, feature_list);
    // Each order is combined with each sigma: 2 orders * 2 sigmas = 4 features
    EXPECT_EQ(config.get_feature_list().size(), 4);
    
    config.set_scaling_mode(scaling_mode_t::both);
    EXPECT_EQ(config.get_scaling_mode(), scaling_mode_t::both);
    
    config.set_overlap_threshold(1e-10);
    EXPECT_DOUBLE_EQ(config.get_overlap_threshold(), 1e-10);
    
    config.set_square(true);
    EXPECT_TRUE(config.get_square());
}

TEST_F(InputTest, read_atom_file) {
    // Test reading a CIF file
    std::unique_ptr<lattice_t<double>> lattice;
    vector<atom_t<double>> atoms;
    atom_type_map_t atom_type_map;
    
    // Use path relative to project root
    std::string cif_path = get_project_path("test/test_files/test.cif");
    read_atom_file<double>(cif_path, lattice, atoms, atom_type_map);
    
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
    vector<gaussian_t<double>> gaussian_table;
    vector<int> offset;

    // Use path relative to project root
    std::string psp_path = get_project_path("test/test_files/test.gpsp");
    read_psp_file<double>(psp_path, atom_type_map, gaussian_table, offset);
    
    // Check if error occurred
    EXPECT_EQ(gmp::gmp_error, gmp::error_t::success);
    
    // Check gaussian_table size
    EXPECT_EQ(gaussian_table.size(), 17); // K, Se, O
    
    // Check if data was loaded for each atom type
    EXPECT_EQ(offset.size(), 4);
}

TEST_F(InputTest, boost_json_parse_arguments) {
    // Read the JSON file
    std::string json_path = get_project_path("test/test_files/input_test.json");
    std::ifstream file(json_path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();

    // Parse JSON using Boost.JSON
    boost::json::error_code ec;
    boost::json::value jv = boost::json::parse(json_str, ec);
    ASSERT_FALSE(ec) << "JSON parsing failed: " << ec.message();

    // Access and verify JSON values
    boost::json::object const& obj = jv.as_object();
    
    // Check file paths
    EXPECT_EQ(obj.at("system file path").as_string(), "test/test_files/test.cif");
    EXPECT_EQ(obj.at("psp file path").as_string(), "test/test_files/test.gpsp");
    EXPECT_EQ(obj.at("output file path").as_string(), "output.dat");
    
    // Check arrays
    boost::json::array const& orders = obj.at("orders").as_array();
    boost::json::array const& sigmas = obj.at("sigmas").as_array();
    
    EXPECT_EQ(orders.size(), 2);
    EXPECT_EQ(sigmas.size(), 2);
    EXPECT_EQ(orders[0].as_int64(), 0);
    EXPECT_EQ(orders[1].as_int64(), 1);
    EXPECT_DOUBLE_EQ(sigmas[0].as_double(), 0.1);
    EXPECT_DOUBLE_EQ(sigmas[1].as_double(), 0.2);
    
    // Check numeric values
    EXPECT_DOUBLE_EQ(obj.at("overlap threshold").as_double(), 1e-10);
    EXPECT_EQ(obj.at("scaling mode").as_int64(), 0);
    EXPECT_EQ(obj.at("square").as_int64(), 1);
    
    // Check array of doubles
    boost::json::array const& tree_min_bounds = obj.at("tree min bounds").as_array();
    EXPECT_EQ(tree_min_bounds.size(), 3);
    EXPECT_DOUBLE_EQ(tree_min_bounds[0].as_double(), 1.0);
    EXPECT_DOUBLE_EQ(tree_min_bounds[1].as_double(), 1.0);
    EXPECT_DOUBLE_EQ(tree_min_bounds[2].as_double(), 1.0);
}

TEST_F(InputTest, to_string_functions) {
    // Test scaling_mode_t to_string
    EXPECT_EQ(gmp_to_string(scaling_mode_t::radial), "radial");
    EXPECT_EQ(gmp_to_string(scaling_mode_t::both), "both");
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 