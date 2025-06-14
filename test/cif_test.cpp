#include <gtest/gtest.h>
#include <gemmi/cif.hpp>
#include <string>
#include <vector>
#include <map>
#include <filesystem>

using namespace gemmi;

#ifndef PROJECT_ROOT
#define PROJECT_ROOT "."
#endif

std::string get_project_path(const std::string& relative_path) {
    return std::filesystem::path(PROJECT_ROOT) / relative_path;
}

TEST(CifTest, ReadCifFile) {

    cif::Document doc =  cif::read_file(get_project_path("test/test_files/test.cif"));

    // Example of using block.find()
    for (cif::Block& block : doc.blocks) {
        std::cout << "Block: " << block.name << "\n\n";
        
        // Find specific tags
        std::vector<std::string> tags = {"_cell_length_a", "_cell_length_b", "_cell_length_c"};
        cif::Table cell_table = block.find(tags);
        
        if (cell_table.ok()) {
            std::cout << "Cell parameters:\n";
            for (const auto& row : cell_table) {
                EXPECT_DOUBLE_EQ(std::stod(row[0]), 8.214313113733247);
                EXPECT_DOUBLE_EQ(std::stod(row[1]), 7.369244);
                EXPECT_DOUBLE_EQ(std::stod(row[2]), 8.908923002714356);
            }
        }

        // Find atom information
        std::vector<std::string> atom_tags = {"label", "type_symbol", "fract_x", "fract_y", "fract_z", "occupancy"};
        cif::Table atom_table = block.find_any("_atom_site_", atom_tags);
        if (atom_table.ok()) {
            std::cout << "\nAtom Information:\n";
            // Print headers
            for (const std::string& tag : atom_table.tags()) {
                std::cout << tag << "\t";
            }
            std::cout << "\n";
            
            // Print values
            for (const auto& row : atom_table) {
                for (size_t i = 0; i < atom_table.width(); ++i) {
                    std::cout << row[i] << "\t";
                }
                std::cout << "\n";
            }
        }
    }
} 