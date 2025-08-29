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

    using gmp::math::array3d_flt;

    // unit_cell_t implementations
    template <typename T>
    inline unit_cell_t<T>::unit_cell_t() : atoms_(), lattice_(), atom_type_map_(), periodicity_{true, true, true} {}

    template <typename T>
    inline unit_cell_t<T>::unit_cell_t(std::string atom_file) : 
        atoms_(), lattice_(), atom_type_map_(), periodicity_{true, true, true} 
    {
        // read atom file 
        read_atom_file<T>(atom_file, lattice_, atoms_, atom_type_map_);

        // set periodicity
        periodicity_ = array3d_bool{true, true, true};

    #ifdef DEBUG
        // dump
        dump();
    #endif
    }

    template <typename T>
    inline const vector<atom_t<T>>& unit_cell_t<T>::get_atoms() const { return atoms_; }

    template <typename T>
    inline const lattice_t<T>* unit_cell_t<T>::get_lattice() const { return lattice_.get(); }

    template <typename T>
    inline const array3d_bool& unit_cell_t<T>::get_periodicity() const { return periodicity_; }

    template <typename T>
    inline const atom_t<T>& unit_cell_t<T>::operator[](size_t i) const { return atoms_[i]; }

    template <typename T>
    inline const atom_type_map_t& unit_cell_t<T>::get_atom_type_map() const { return atom_type_map_; }

    template <typename T>
    inline void unit_cell_t<T>::set_lattice(std::unique_ptr<lattice_t<T>>&& lattice) { lattice_ = std::move(lattice); }

    template <typename T>
    inline void unit_cell_t<T>::set_atoms(vector<atom_t<T>>&& atoms) { atoms_ = std::move(atoms); }

    template <typename T>
    inline void unit_cell_t<T>::set_atom_type_map(atom_type_map_t&& atom_type_map) { atom_type_map_ = std::move(atom_type_map); }

    template <typename T>
    inline void unit_cell_t<T>::set_periodicity(const array3d_bool& periodicity) { periodicity_ = periodicity; }

    template <typename T>
    inline void unit_cell_t<T>::dump() const {
        std::cout << "Lattice: " << std::endl;
        lattice_->dump();
        std::cout << "Atom type map: " << std::endl;
        for (const auto& [type, id] : atom_type_map_) {
            std::cout << "Type: " << type << ", ID: " << static_cast<int>(id) << std::endl;
        }
        std::cout << "Atoms: " << std::endl;
        for (const auto& atom : atoms_) {
            std::cout << "Atom: " << static_cast<int>(atom.type_id) << ", Position: " << atom.pos << ", Occupancy: " << atom.occ << std::endl;            
        }        
    }

    // Explicit instantiations for unit_cell_t
    template class unit_cell_t<float>;
    template class unit_cell_t<double>;

    // psp_config_t implementations
    template <typename T>
    inline psp_config_t<T>::psp_config_t(std::string psp_file, const unit_cell_t<T>* unit_cell) : 
        gaussian_table_(), offset_() 
    {
        // read psp file 
        read_psp_file<T>(psp_file, unit_cell->get_atom_type_map(), gaussian_table_, offset_);

    #ifdef DEBUG
        dump();
    #endif
    }

    template <typename T>
    inline const vector<int>& psp_config_t<T>::get_offset() const { return offset_; }

    template <typename T>
    inline const typename psp_config_t<T>::gaussian_type& psp_config_t<T>::operator[](const int idx) const { return gaussian_table_[idx]; }

    template <typename T>
    inline void psp_config_t<T>::dump() const 
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

    // Explicit instantiations for psp_config_t
    template class psp_config_t<float>;
    template class psp_config_t<double>;

    // Explicit instantiations for template member functions  
    // template psp_config_t<float>::psp_config_t(std::string, const unit_cell_t<float>*);
    // template psp_config_t<double>::psp_config_t(std::string, const unit_cell_t<double>*);

    // read cif file 
    template <typename T>
    inline void read_atom_file(const std::string& atom_file, std::unique_ptr<lattice_t<T>>& lattice, vector<atom_t<T>>& atoms, atom_type_map_t& atom_type_map) 
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
        gmp::math::array3d_t<T> cell = {
            static_cast<T>(std::stod(cell_table[0][0])), static_cast<T>(std::stod(cell_table[0][1])), static_cast<T>(std::stod(cell_table[0][2]))
        };

        tags = {"_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"};
        gemmi::cif::Table angle_table = block.find(tags);
        gmp::math::array3d_t<T> angle = {
            static_cast<T>(std::stod(angle_table[0][0])), static_cast<T>(std::stod(angle_table[0][1])), static_cast<T>(std::stod(angle_table[0][2]))
        };

        // create lattice from cell and angle
        lattice = gmp::geometry::cell_info_to_lattice<T>(cell, angle);

        // get atom information
        atoms.clear();
        atom_type_map.clear();

        std::vector<std::string> atom_tags = {"type_symbol", "fract_x", "fract_y", "fract_z", "occupancy"};
        gemmi::cif::Table atom_table = block.find_any("_atom_site_", atom_tags);

        for (const auto& row : atom_table) {
            std::string type_symbol = row[0];
            T fract_x = static_cast<T>(std::stod(row[1]));
            T fract_y = static_cast<T>(std::stod(row[2]));
            T fract_z = static_cast<T>(std::stod(row[3]));
            T occupancy = static_cast<T>(std::stod(row[4]));
            if (atom_type_map.find(type_symbol) == atom_type_map.end()) {
                atom_type_map[type_symbol] = static_cast<uint32_t>(atom_type_map.size());
            }
            atoms.push_back({point3d_t<T>{fract_x, fract_y, fract_z}, occupancy, atom_type_map[type_symbol]});
        }

        return;
    }

    // Explicit instantiations for read_atom_file
    template void read_atom_file<float>(const std::string&, std::unique_ptr<lattice_t<float>>&, vector<atom_t<float>>&, atom_type_map_t&);
    template void read_atom_file<double>(const std::string&, std::unique_ptr<lattice_t<double>>&, vector<atom_t<double>>&, atom_type_map_t&);

    // read psp file 
    template <typename T>
    inline void read_psp_file(const std::string& psp_file, const atom_type_map_t& atom_type_map, vector<gaussian_t<T>>& gaussian_table, vector<int>& offset) 
    {
        std::vector<std::vector<gaussian_t<T>>> gaussian_table_2d(atom_type_map.size());

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
                        T B, beta;
                        iss >> B >> beta;
                        gaussian_table_2d[id].push_back({B, beta});
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

    // Explicit instantiations for read_psp_file
    template void read_psp_file<float>(const std::string&, const atom_type_map_t&, vector<gaussian_t<float>>&, vector<int>&);
    template void read_psp_file<double>(const std::string&, const atom_type_map_t&, vector<gaussian_t<double>>&, vector<int>&);

    template <typename T>
    inline vector<point3d_t<T>> set_ref_positions(const array3d_int32& ref_grid, const vector<atom_t<T>>& atoms)
    {
        vector<point3d_t<T>> ref_positions;
        if (ref_grid[0] <= 0 || ref_grid[1] <= 0 || ref_grid[2] <= 0) {
            for (const auto& atom : atoms) {
                ref_positions.emplace_back(atom.pos);
            }
        } else {
            ref_positions.reserve(ref_grid[0] * ref_grid[1] * ref_grid[2]);
            for (auto k = 0; k < ref_grid[2]; ++k) {
                for (auto j = 0; j < ref_grid[1]; ++j) {
                    for (auto i = 0; i < ref_grid[0]; ++i) {
                        ref_positions.push_back(point3d_t<T>{
                            static_cast<T>(i) / static_cast<T>(ref_grid[0]), 
                            static_cast<T>(j) / static_cast<T>(ref_grid[1]), 
                            static_cast<T>(k) / static_cast<T>(ref_grid[2])
                        });
                    }
                }
            }
        }
        return ref_positions;
    }

    // Explicit instantiations for set_ref_positions
    template vector<point3d_t<float>> set_ref_positions<float>(const array3d_int32&, const vector<atom_t<float>>&);
    template vector<point3d_t<double>> set_ref_positions<double>(const array3d_int32&, const vector<atom_t<double>>&);

}} 