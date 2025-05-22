#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include "input.hpp"
#include "error.hpp"
#include "util.hpp"

namespace gmp { namespace input {

    using namespace gmp;

    // file path config
    void file_path_t::print_config() const 
    {
        std::cout << "Atom file: " << atom_file_ << std::endl;
        std::cout << "PSP file: " << psp_file_ << std::endl;
        std::cout << "Output file: " << output_file_ << std::endl;
    }

    // descriptor config
    void descriptor_config_t::set_feature_list(std::vector<int> orders, std::vector<double> sigmas, std::vector<std::tuple<int, double>> feature_list) 
    {
        feature_list_.clear();

        // parse feature list
        if (!feature_list.empty()) {
            for (auto &feature : feature_list) {
                if (std::get<0>(feature) < -1 || std::get<0>(feature) > 9) {
                    update_error(gmp::error_t::invalid_order_sigma); return;
                } else if (std::get<0>(feature) == -1) {
                    feature_list_.push_back(feature_t(-1, 0.0));
                } else {
                    feature_list_.push_back(feature_t(std::get<0>(feature), std::get<1>(feature)));
                }
            }
        } else {
            for (auto order : orders) {
                for (auto sigma : sigmas) {
                    if (order < -1 || order > 9) {
                        update_error(gmp::error_t::invalid_order_sigma); return;
                    } else if (order == -1) {
                        feature_list_.push_back(feature_t(-1, 0.0));
                    } else {
                        feature_list_.push_back(feature_t(order, sigma));
                    }
                }
            }
        }

        if (feature_list_.empty()) {
            update_error(gmp::error_t::invalid_feature_list);
            return;
        }

        // sort the feature list based on sigma, then order 
        std::sort(feature_list_.begin(), feature_list_.end());
    }

    void descriptor_config_t::print_config() const 
    {
        std::cout << "Feature list: " << std::endl;
        for (const auto& feature : feature_list_) {
            std::cout << "  Order: " << feature.order << ", Sigma: " << feature.sigma << std::endl;
        }
        std::cout << "Cutoff method: " << gmp_to_string(cutoff_method_) << std::endl;
        std::cout << "Cutoff: " << cutoff_ << std::endl;
        std::cout << "Overlap threshold: " << overlap_threshold_ << std::endl;
        std::cout << "Scaling mode: " << gmp_to_string(scaling_mode_) << std::endl;
    }

    // read psp file 
    void read_psp_file(const std::string& psp_file, const atom_type_map_t& atom_type_map, vec<vec<gaussian_t>>& gaussian_table) 
    {
        gaussian_table.resize(atom_type_map.size());

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
                        gaussian_table[id].push_back(gaussian_t{B, beta});
                    }
                }
            }
        }

        file.close();
        return;
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

    // psp config 
    int psp_config_t::get_num_gaussians_offset(vec<int>& offset) const 
    {
        auto num_atoms = gaussian_table_.size();
        offset.resize(num_atoms);
        offset[0] = 0;
        for (size_t i = 1; i < num_atoms; ++i) {
            offset[i] = offset[i-1] + gaussian_table_[i-1].size();
        }
        return offset.back() + gaussian_table_.back().size();
    }

    void psp_config_t::print_config() const 
    {
        std::cout << "Gaussian table: " << std::endl;
        for (size_t i = 0; i < gaussian_table_.size(); ++i) {
            const auto& gaussian = gaussian_table_[i];
            std::cout << "Atom type: " << i << std::endl;
            for (const auto& g : gaussian) {
                std::cout << "Gaussian: " << g.B << ", Beta: " << g.beta << std::endl;
            }
        }
    }

    // parse arguments 
    void input_t::parse_arguments(int argc, char* argv[]) 
    {
        if (argc == 2 && std::string(argv[1]) == "-h") {
            print_help();
            return;
        }
        if (argc % 2 == 0) {
            std::cerr << "Error in command line." << std::endl;
            gmp_error = error_t::invalid_argument;
            return;
        }
        
        std::vector<int> orders;
        std::vector<double> sigmas;
        std::vector<std::tuple<int, double>> feature_list;
        
        files = std::make_unique<file_path_t>();
        descriptor_config = std::make_unique<descriptor_config_t>();

        for (int idx = 1; idx < argc; idx += 2) {
            const std::string &key = argv[idx];
            const std::string &val = argv[idx+1];
            if (key == "orders") {
                orders = util::parse_line_pattern<int>(val);
            } else if (key == "sigmas") {
                sigmas = util::parse_line_pattern<double>(val);
            } else if (key == "featureList") {
                feature_list = util::parse_line_pattern<int, double>(val);
            } else if (key == "systemPath") {
                this->files->set_atom_file(val);
            } else if (key == "pspPath") {
                this->files->set_psp_file(val);
            } else if (key == "square") {
                this->descriptor_config->set_square(std::stoi(val));
            } else if (key == "cutoffMethod") {
                switch (std::stoi(val))
                {
                case 0:
                    this->descriptor_config->set_cutoff_method(cutoff_method_t::custom_cutoff);
                    break;
                case 1:
                    this->descriptor_config->set_cutoff_method(cutoff_method_t::cutoff_sigma);
                    break;
                case 2:
                    this->descriptor_config->set_cutoff_method(cutoff_method_t::cutoff_sigma_elemental);
                    break;
                case 3:
                    this->descriptor_config->set_cutoff_method(cutoff_method_t::cutoff_feature_elemental);
                    break;
                case 4:
                    this->descriptor_config->set_cutoff_method(cutoff_method_t::cutoff_feature_gaussian);
                    break;
                default:                    
                    update_error(gmp::error_t::invalid_cutoff_method);
                    return;
                }
            } else if (key == "cutoff") {
                this->descriptor_config->set_cutoff(std::stod(val));
            } else if (key == "overlapThreshold") {
                this->descriptor_config->set_overlap_threshold(std::stod(val));
            } else if (key == "scalingMode") {
                switch (std::stoi(val))
                {
                case 0:
                    this->descriptor_config->set_scaling_mode(scaling_mode_t::radial);
                    break;
                case 1:
                    this->descriptor_config->set_scaling_mode(scaling_mode_t::both);
                    break;
                default:
                    update_error(gmp::error_t::invalid_scaling_mode);
                    return;
                }
            } else if (key == "outputPath") {
                this->files->set_output_file(val);
            } else {
                update_error(gmp::error_t::invalid_argument);
                return;
            }
        }
        this->descriptor_config->set_feature_list(orders, sigmas, feature_list);
        return;
    }

    void input_t::print_help() const 
    {
        std::cout << "Usage: gmp_featurizer [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  systemPath <path>          Path to the system file (CIF format)" << std::endl;
        std::cout << "  pspPath <path>             Path to the pseudopotential file" << std::endl;
        std::cout << "  orders <list>              List of orders (e.g., -1,0,1,2)" << std::endl;
        std::cout << "  sigmas <list>              List of sigmas (e.g., 0.1,0.2,0.3)" << std::endl;
        std::cout << "  featureList <list>         List of feature pairs (e.g., (1,0.1),(2,0.2))" << std::endl;
        std::cout << "  square <int>               Square option (0 or 1)" << std::endl;
        std::cout << "  cutoffMethod <int>         Cutoff method (0 to 4)" << std::endl;
        std::cout << "  cutoff <double>            Cutoff value" << std::endl;
        std::cout << "  overlapThreshold <double>  Overlap threshold" << std::endl;
        std::cout << "  scalingMode <int>          Scaling mode (0 for radial, 1 for both)" << std::endl;
        std::cout << "  outputPath <path>          Path to the output file" << std::endl;
        std::cout << "  -h                         Print this help message" << std::endl;
        return;
    }

    void input_t::print_config() const 
    {
        if (files) files->print_config();
        if (descriptor_config) descriptor_config->print_config();
    }

}}