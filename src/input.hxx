#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>
#include "error.hpp"
#include "util.hpp"

namespace gmp { namespace input {

    using namespace gmp;

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

    // Type aliases using configured floating-point type
    using descriptor_config_flt = descriptor_config_t<gmp::gmp_float>;
    using feature_flt = feature_t<gmp::gmp_float>;

}} 