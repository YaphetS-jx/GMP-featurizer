#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <boost/json/src.hpp>

#include "input.hpp"
#include "error.hpp"
#include "util.hpp"

namespace gmp { namespace input {

    using namespace gmp;

    input_t::input_t(const std::string& json_file) : 
        files(std::make_unique<file_path_t>()),
        descriptor_config(std::make_unique<descriptor_config_t>()) 
    {
        if (json_file == "-h") {
            print_help();
            exit(0);
        }

        // parse JSON file
        parse_json(json_file);

        // print config
    #ifdef DEBUG
        dump();
    #endif
    }

    // file path config
    void file_path_t::dump() const 
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
        bool negative_order = false;
        if (!feature_list.empty()) {
            for (auto &feature : feature_list) {
                if (std::get<0>(feature) < -1 || std::get<0>(feature) > 9) {
                    update_error(gmp::error_t::invalid_order_sigma); return;
                } else if (std::get<0>(feature) == -1) {
                    if (!negative_order) feature_list_.push_back(feature_t(-1, 0.0));
                    negative_order = true;
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
                        if (!negative_order) feature_list_.push_back(feature_t(-1, 0.0));
                        negative_order = true;
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

    void descriptor_config_t::dump() const 
    {
        std::cout << "Feature list: " << std::endl;
        for (const auto& feature : feature_list_) {
            std::cout << "  Order: " << feature.order << ", Sigma: " << feature.sigma << std::endl;
        }
        std::cout << "Overlap threshold: " << overlap_threshold_ << std::endl;
        std::cout << "Scaling mode: " << gmp_to_string(scaling_mode_) << std::endl;
        std::cout << "Ref grid: " << ref_grid_[0] << ", " << ref_grid_[1] << ", " << ref_grid_[2] << std::endl;
    }

    // parse JSON  
    void input_t::parse_json(const std::string& json_file) 
    {
        std::ifstream inFile(json_file);
        if (!inFile.is_open()) {
            update_error(gmp::error_t::invalid_json_file);
            return;
        }

        std::stringstream buffer;
        buffer << inFile.rdbuf();
        std::string json_str = buffer.str();

        boost::json::error_code ec;
        boost::json::value jv = boost::json::parse(json_str, ec);
        if (ec) {
            update_error(gmp::error_t::invalid_json_file);
            return;
        }

        boost::json::object const& config = jv.as_object();
        
        std::vector<int> orders;
        std::vector<double> sigmas;
        std::vector<std::tuple<int, double>> feature_list;

        // Required entries
        this->files->set_atom_file(std::string(config.at("system file path").as_string()));
        this->files->set_psp_file(std::string(config.at("psp file path").as_string()));
        this->files->set_output_file(std::string(config.at("output file path").as_string()));
        this->descriptor_config->set_square(config.at("square").as_int64());
        this->descriptor_config->set_overlap_threshold(config.at("overlap threshold").as_double());
        this->descriptor_config->set_scaling_mode(static_cast<scaling_mode_t>(config.at("scaling mode").as_int64()));
        
        if (config.contains("ref_grid")) {
            auto const& ref_grid_json = config.at("ref_grid").as_array();
            if (ref_grid_json.size() == 3) {
                array3d_int32 ref_grid_array(
                    ref_grid_json[0].as_int64(),
                    ref_grid_json[1].as_int64(),
                    ref_grid_json[2].as_int64()
                );
                this->descriptor_config->set_ref_grid(ref_grid_array);
            }
        }

        // Optional entries
        if (config.contains("orders")) {
            auto const& orders_json = config.at("orders").as_array();
            for (auto const& val : orders_json) {
                orders.push_back(val.as_int64());
            }
        }
        if (config.contains("sigmas")) {
            auto const& sigmas_json = config.at("sigmas").as_array();
            for (auto const& val : sigmas_json) {
                sigmas.push_back(val.as_double());
            }
        }
        if (config.contains("feature lists")) {
            auto const& feature_lists_json = config.at("feature lists").as_array();
            for (auto const& pair : feature_lists_json) {
                auto const& pair_array = pair.as_array();
                feature_list.emplace_back(pair_array[0].as_int64(), pair_array[1].as_double());
            }
        }
        
        this->descriptor_config->set_feature_list(orders, sigmas, feature_list);        
        return;
    }

    void input_t::print_help() const 
    {   
        std::cout << "Please provide the path to the JSON file." << std::endl;
        std::cout << "Usage: gmp_featurizer [options] in JSON File. " << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  system file path <path>          Path to the system file (CIF format)" << std::endl;
        std::cout << "  psp file path <path>             Path to the pseudopotential file" << std::endl;
        std::cout << "  orders <list>                    List of orders (e.g., -1,0,1,2)" << std::endl;
        std::cout << "  sigmas <list>                    List of sigmas (e.g., 0.1,0.2,0.3)" << std::endl;
        std::cout << "  feature lists <list>             List of feature pairs (e.g., (1,0.1),(2,0.2))" << std::endl;
        std::cout << "  square <int>                     Square option (0 or 1)" << std::endl;
        std::cout << "  overlap threshold <double>       Overlap threshold" << std::endl;
        std::cout << "  scaling mode <int>               Scaling mode (0 for radial, 1 for both)" << std::endl;
        std::cout << "  output file path <path>          Path to the output file" << std::endl;
        std::cout << "  ref_grid <list>                  Reference grid (e.g., 10,10,10)" << std::endl;
        std::cout << "  -h                               Print this help message" << std::endl;
        return;
    }

    void input_t::dump() const 
    {
        if (files) files->dump();
        if (descriptor_config) descriptor_config->dump();
    }

}}