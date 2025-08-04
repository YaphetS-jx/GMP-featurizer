#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include <nlohmann/json.hpp>

#include "input.hpp"
#include "error.hpp"
#include "util.hpp"

namespace gmp { namespace input {

    using namespace gmp;
    using json = nlohmann::json;

    input_t::input_t(const std::string& json_file) : 
        files(std::make_unique<file_path_t>()),
        descriptor_config(std::make_unique<descriptor_config_flt>()) 
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

        json jv;
        try {
            jv = json::parse(json_str);
        } catch (json::parse_error& e) {
            update_error(gmp::error_t::invalid_json_file);
            return;
        }

        json const& config = jv;
        
        std::vector<int> orders;
        std::vector<gmp_float> sigmas;
        std::vector<std::tuple<int, gmp_float>> feature_list;

        // Required entries
        this->files->set_atom_file(config["system file path"].get<std::string>());
        this->files->set_psp_file(config["psp file path"].get<std::string>());
        this->files->set_output_file(config["output file path"].get<std::string>());

        // Optional entries
        if (config.contains("square")) {
            this->descriptor_config->set_square(config["square"].get<int32_t>());
        }
        if (config.contains("overlap threshold")) {
            this->descriptor_config->set_overlap_threshold(static_cast<gmp_float>(config["overlap threshold"].get<double>()));
        }
        if (config.contains("scaling mode")) {
            this->descriptor_config->set_scaling_mode(static_cast<scaling_mode_t>(config["scaling mode"].get<int32_t>()));
        }
        
        if (config.contains("reference grid")) {
            json const& ref_grid_json = config["reference grid"];
            if (ref_grid_json.size() == 3) {
                array3d_int32 ref_grid_array{
                    ref_grid_json[0].get<int32_t>(),
                    ref_grid_json[1].get<int32_t>(),
                    ref_grid_json[2].get<int32_t>()
                };
                this->descriptor_config->set_ref_grid(ref_grid_array);
            }
        }
        
        if (config.contains("num bits per dim")) {
            this->descriptor_config->set_num_bits_per_dim(static_cast<uint8_t>(config["num bits per dim"].get<int32_t>()));
        }
        if (config.contains("num threads")) {
            this->descriptor_config->set_num_threads(static_cast<size_t>(config["num threads"].get<int32_t>()));
        }

        if (config.contains("orders")) {
            json const& orders_json = config["orders"];
            for (auto const& val : orders_json) {
                orders.push_back(val.get<int32_t>());
            }
        }
        if (config.contains("sigmas")) {
            json const& sigmas_json = config["sigmas"];
            for (auto const& val : sigmas_json) {
                sigmas.push_back(static_cast<gmp_float>(val.get<double>()));
            }
        }
        if (config.contains("feature lists")) {
            json const& feature_lists_json = config["feature lists"];
            for (auto const& pair : feature_lists_json) {
                json const& pair_array = pair;
                feature_list.emplace_back(pair_array[0].get<int32_t>(), static_cast<gmp_float>(pair_array[1].get<double>()));
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
        std::cout << "  reference grid <list>            Reference grid (e.g., 10,10,10)" << std::endl;
        std::cout << "  num bits per dim <int>           Number of bits per dimension" << std::endl;
        std::cout << "  num threads <int>                Number of threads" << std::endl;
        std::cout << "  -h                               Print this help message" << std::endl;
        return;
    }

    void input_t::dump() const 
    {
        if (files) files->dump();
        if (descriptor_config) descriptor_config->dump();
    }

    // descriptor_config_t template implementations
    template <typename T>
    void descriptor_config_t<T>::set_feature_list(std::vector<int> orders, std::vector<T> sigmas, std::vector<std::tuple<int, T>> feature_list) 
    {
        feature_list_.clear();

        // parse feature list
        bool negative_order = false;
        if (!feature_list.empty()) {
            for (auto &feature : feature_list) {
                if (std::get<0>(feature) < -1 || std::get<0>(feature) > 9) {
                    update_error(gmp::error_t::invalid_order_sigma); return;
                } else if (std::get<0>(feature) == -1) {
                    if (!negative_order) feature_list_.push_back(feature_t<T>(-1, 0.0));
                    negative_order = true;
                } else {
                    feature_list_.push_back(feature_t<T>(std::get<0>(feature), std::get<1>(feature)));
                }
            }
        } else {
            for (auto order : orders) {
                for (auto sigma : sigmas) {
                    if (order < -1 || order > 9) {
                        update_error(gmp::error_t::invalid_order_sigma); return;
                    } else if (order == -1) {
                        if (!negative_order) feature_list_.push_back(feature_t<T>(-1, 0.0));
                        negative_order = true;
                    } else {
                        feature_list_.push_back(feature_t<T>(order, sigma));
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

    template <typename T>
    void descriptor_config_t<T>::dump() const 
    {
        std::cout << "Feature list: " << std::endl;
        for (const auto& feature : feature_list_) {
            std::cout << "  Order: " << feature.order << ", Sigma: " << feature.sigma << std::endl;
        }
        std::cout << "Overlap threshold: " << overlap_threshold_ << std::endl;
        std::cout << "Scaling mode: " << gmp_to_string(scaling_mode_) << std::endl;
        std::cout << "Ref grid: " << ref_grid_[0] << ", " << ref_grid_[1] << ", " << ref_grid_[2] << std::endl;
    }

    // Explicit instantiations for descriptor_config_t (used externally)
    template class descriptor_config_t<float>;
    template class descriptor_config_t<double>;

}}