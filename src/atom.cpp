#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#include "atom.hpp"
#include "error.hpp"
#include "math.hpp"
#include "geometry.hpp"
#include "util.hpp"


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
        std::ifstream file(atom_file);
        if (!file.is_open()) {
            update_error(gmp::error_t::invalid_atom_file);
            return;
        }

        atoms.clear();
        atom_type_map.clear();

        std::string line;
        bool read_atoms = false;
        std::vector<std::string> atom_columns;
        array3d_flt64 cell;
        array3d_flt64 angle;

        while (getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            // Part 1: Read geometry information
            if (line.find("_cell_length_a") != std::string::npos)
                cell[0] = stod(line.substr(line.find_last_of(" ") + 1));
            else if (line.find("_cell_length_b") != std::string::npos)
                cell[1] = stod(line.substr(line.find_last_of(" ") + 1));
            else if (line.find("_cell_length_c") != std::string::npos)
                cell[2] = stod(line.substr(line.find_last_of(" ") + 1));
            else if (line.find("_cell_angle_alpha") != std::string::npos)
                angle[0] = stod(line.substr(line.find_last_of(" ") + 1));
            else if (line.find("_cell_angle_beta") != std::string::npos)
                angle[1] = stod(line.substr(line.find_last_of(" ") + 1));
            else if (line.find("_cell_angle_gamma") != std::string::npos)
                angle[2] = stod(line.substr(line.find_last_of(" ") + 1));

            // Part 2: Read atom positions
            else if (line.find("loop_") != std::string::npos) {
                // if already finished reading atoms, break
                if (atom_columns.size())  break;
                // Start of a data loop, reset atom_columns
                atom_columns.clear();
                read_atoms = true;
            } else if (read_atoms && !line.empty()) {
                if (line.compare(0, 11, "_atom_site_") == 0) {
                    gmp::util::trim(line);
                    atom_columns.push_back(line);
                    continue;
                } 
                // if no information provided yet, skip the line
                if (!atom_columns.size()) continue;

                std::istringstream data_line(line);
                std::string value;
                point_flt64 position;
                double occupancy = 1.0;
                std::string type;
                for (size_t i = 0; i < atom_columns.size(); ++i) {
                    data_line >> value;
                    if (atom_columns[i] == "_atom_site_label") {
                        type = value.substr(0, value.find_first_of("0123456789")); // remove index 
                    } else if (atom_columns[i] == "_atom_site_fract_x") {
                        position.x = std::stod(value);
                    } else if (atom_columns[i] == "_atom_site_fract_y") {
                        position.y = std::stod(value);
                    } else if (atom_columns[i] == "_atom_site_fract_z") {
                        position.z = std::stod(value);
                    } else if (atom_columns[i] == "_atom_site_occupancy") {
                        occupancy = std::stod(value);
                    }
                }
                if (atom_type_map.find(type) == atom_type_map.end()) {
                    atom_type_map[type] = static_cast<atom_type_id_t>(atom_type_map.size());
                }
                // round between 0 and 1
                position.x = gmp::math::round_to_0_1(position.x);
                position.y = gmp::math::round_to_0_1(position.y);
                position.z = gmp::math::round_to_0_1(position.z);
                // make sure it is fractional coordinates
                atoms.emplace_back(position, occupancy, atom_type_map[type]);
            }
        }

        file.close();
        // set lattice 
        lattice = gmp::geometry::cell_info_to_lattice(cell, angle);
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

    vec<point_flt64> set_ref_positions(const unit_cell_t* unit_cell)
    {
        vec<point_flt64> ref_positions;
        for (const auto& atom : unit_cell->get_atoms()) {
            ref_positions.emplace_back(atom.pos());
        }
        return ref_positions;
    }

} }