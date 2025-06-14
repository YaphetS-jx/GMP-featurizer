#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#include "atom.hpp"
#include "error.hpp"
#include "math.hpp"
#include "geometry.hpp"
#include "util.hpp"
#include <gemmi/cif.hpp>

namespace gmp { namespace atom {

    unit_cell_t::unit_cell_t(std::string atom_file) : 
        atoms_(), lattice_(), atom_type_map_(), periodicity_{true, true, true} 
    {
        // read atom file 
        read_atom_file(atom_file, lattice_, atoms_, atom_type_map_);

        // set periodicity
        periodicity_ = array3d_bool{true, true, true};

        // dump
        lattice_->dump();
    }

    void unit_cell_t::dump() const {
        std::cout << "Lattice: " << std::endl;
        lattice_->dump();
        std::cout << "Atom type map: " << std::endl;
        for (const auto& [type, id] : atom_type_map_) {
            std::cout << "Type: " << type << ", ID: " << static_cast<int>(id) << std::endl;
        }
        std::cout << "Atoms: " << std::endl;
        for (const auto& atom : atoms_) {
            std::cout << "Atom: " << static_cast<int>(atom.id()) << ", Position: " << atom.pos() << ", Occupancy: " << atom.occ() << std::endl;            
        }        
    }

    // read cif file 
    void read_atom_file(const std::string& atom_file, std::unique_ptr<lattice_t>& lattice, vec<atom_t>& atoms, atom_type_map_t& atom_type_map) 
    {
        gemmi::cif::Document doc;
        try {
            doc = gemmi::cif::read_file(atom_file);
        } catch (const std::exception& e) {
            update_error(gmp::error_t::invalid_atom_file);
            return;
        }

        gemmi::cif::Block block = doc.blocks[0];
        assert(doc.blocks.size() == 1);
        
        std::vector<std::string> tags = {"_cell_length_a", "_cell_length_b", "_cell_length_c"};
        gemmi::cif::Table cell_table = block.find(tags);
        array3d_flt64 cell = {
            std::stod(cell_table[0][0]), std::stod(cell_table[0][1]), std::stod(cell_table[0][2])
        };

        tags = {"_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"};
        gemmi::cif::Table angle_table = block.find(tags);
        array3d_flt64 angle = {
            std::stod(angle_table[0][0]), std::stod(angle_table[0][1]), std::stod(angle_table[0][2])
        };

        // create lattice from cell and angle
        lattice = gmp::geometry::cell_info_to_lattice(cell, angle);

        // get atom information
        atoms.clear();
        atom_type_map.clear();

        std::vector<std::string> atom_tags = {"type_symbol", "fract_x", "fract_y", "fract_z", "occupancy"};
        gemmi::cif::Table atom_table = block.find_any("_atom_site_", atom_tags);

        for (const auto& row : atom_table) {
            std::string type_symbol = row[0];
            double fract_x = std::stod(row[1]);
            double fract_y = std::stod(row[2]);
            double fract_z = std::stod(row[3]);
            double occupancy = std::stod(row[4]);
            if (atom_type_map.find(type_symbol) == atom_type_map.end()) {
                atom_type_map[type_symbol] = static_cast<atom_type_id_t>(atom_type_map.size());
            }
            atoms.emplace_back(point_flt64{fract_x, fract_y, fract_z}, occupancy, atom_type_map[type_symbol]);
        }

        return;
    }

    // read psp file 
    void read_psp_file(const std::string& psp_file, const atom_type_map_t& atom_type_map, vec<gaussian_t>& gaussian_table, vec<int>& offset) 
    {
        vec<vec<gaussian_t>> gaussian_table_2d(atom_type_map.size());

        std::ifstream file(psp_file);
        if (!file.is_open()) {
            update_error(gmp::error_t::invalid_psp_file);
            return;
        }
        std::string line;
        int num_elements = 0;

        while (getline(file, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') continue; // Skip comment lines
            if (line[0] == '!') {
                // Read the number of elements
                std::istringstream iss(line.substr(1));
                iss >> num_elements;
            } else if (line[0] == '*') {
                // Skip rows starting with '*'
                continue;
            } else {
                std::istringstream iss(line);
                std::string atom_type;
                int atom_index, num_rows;

                // Read atom type, index, and number of rows
                iss >> atom_type >> atom_index >> num_rows;
                if (atom_type_map.find(atom_type) == atom_type_map.end()) {
                    // Skip the next numRows rows if atom type is not in the set
                    for (int i = 0; i < num_rows; ++i) {
                        getline(file, line);
                    }
                } else {
                    // auto id = atom_type_map[atom_type];
                    if (atom_type_map.find(atom_type) == atom_type_map.end()) {
                        update_error(gmp::error_t::missing_atom_psp);
                        return;
                    }
                    auto id = atom_type_map.at(atom_type);
                    // Read the matrix data for the atom
                    for (int i = 0; i < num_rows; ++i) {
                        getline(file, line);
                        iss.clear();  // Clear any error flags
                        iss.str(line);
                        double B, beta;
                        iss >> B >> beta;
                        gaussian_table_2d[id].emplace_back(B, beta);
                    }
                }
            }
        }

        file.close();

        // flatten the 2d vector
        for (const auto& gaussians : gaussian_table_2d) {
            gaussian_table.insert(gaussian_table.end(), gaussians.begin(), gaussians.end());
        }

        // calculate the offset
        offset.resize(atom_type_map.size() + 1);
        offset[0] = 0;
        for (size_t i = 1; i < atom_type_map.size() + 1; ++i) {
            offset[i] = offset[i-1] + gaussian_table_2d[i-1].size();
        }
        return;
    }

    // psp config 
    psp_config_t::psp_config_t(std::string psp_file, const unit_cell_t* unit_cell) : 
        gaussian_table_(), offset_() 
    {
        // read psp file 
        read_psp_file(psp_file, unit_cell->get_atom_type_map(), gaussian_table_, offset_);
    }


    void psp_config_t::dump() const 
    {
        std::cout << "Gaussian table: " << std::endl;
        for (auto i = 0; i < offset_.size() - 1; ++i) {
            std::cout << "Atom type: " << i << std::endl;
            for (auto j = offset_[i]; j < offset_[i+1]; ++j) {
                std::cout << "Gaussian: " << gaussian_table_[j].B << ", Beta: " << gaussian_table_[j].beta << std::endl;
            }
        }
        std::cout << "Offset: " << std::endl;
        for (auto i = 0; i < offset_.size(); ++i) {
            std::cout << offset_[i] << ", ";
        }
        std::cout << std::endl;
    }

    vec<point_flt64> set_ref_positions(const array3d_int32& ref_grid, const vec<atom_t>& atoms)
    {
        vec<point_flt64> ref_positions;
        if (ref_grid[0] <= 0 || ref_grid[1] <= 0 || ref_grid[2] <= 0) {
            for (const auto& atom : atoms) {
                ref_positions.emplace_back(atom.pos());
            }
        } else {
            ref_positions.reserve(ref_grid[0] * ref_grid[1] * ref_grid[2]);
            for (auto k = 0; k < ref_grid[2]; ++k) {
                for (auto j = 0; j < ref_grid[1]; ++j) {
                    for (auto i = 0; i < ref_grid[0]; ++i) {
                        ref_positions.push_back(point_flt64{
                            static_cast<double>(i) / static_cast<double>(ref_grid[0]), 
                            static_cast<double>(j) / static_cast<double>(ref_grid[1]), 
                            static_cast<double>(k) / static_cast<double>(ref_grid[2])
                        });
                    }
                }
            }
        }
        return ref_positions;
    }

} }