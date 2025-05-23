#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include "input.hpp"
#include "error.hpp"
#include "util.hpp"

namespace gmp { namespace input {

    using namespace gmp;

    input_t::input_t(int argc, char* argv[]) : 
        files(std::make_unique<file_path_t>()),
        descriptor_config(std::make_unique<descriptor_config_t>()) 
    {
        // parse arguments
        parse_arguments(argc, argv);

        // print config
        print_config();
    }

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