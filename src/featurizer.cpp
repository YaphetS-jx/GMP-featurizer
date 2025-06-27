#include <cmath>
#include <iostream>
#include <future>
#include "featurizer.hpp"
#include "error.hpp"
#include "math.hpp"
#include "input.hpp"
#include "types.hpp"
#include "util.hpp"
#include "region_query.hpp"
// #include "mcsh_simd.hpp"
#include "resources.hpp"

namespace gmp { namespace featurizer {

    // kernel_params_table_t ctor
    kernel_params_table_t::kernel_params_table_t(const descriptor_config_t* descriptor_config, const psp_config_t* psp_config) 
        : table_(), num_gaussians_(psp_config->get_offset().back()), num_features_(descriptor_config->get_feature_list().size()) 
    {
        table_.reserve(num_features_ * num_gaussians_);
        
        auto feature_list = descriptor_config->get_feature_list();
        auto scaling_mode = descriptor_config->get_scaling_mode();
        auto num_atom_types = psp_config->get_offset().size() - 1;

        for (const auto& feature : feature_list) {
            const double A = get_scaling_const(feature.order, feature.sigma, scaling_mode);
            const double alpha = (feature.order == -1) ? 0. : (1.0 / (2.0 * feature.sigma * feature.sigma));
            for (atom_type_id_t atom_type_idx = 0; atom_type_idx < num_atom_types; ++atom_type_idx) {
                // get the ranges of gaussians for the current atom type
                auto start = psp_config->get_offset()[atom_type_idx];
                auto end = psp_config->get_offset()[atom_type_idx + 1];
                // loop over the gaussians for the current atom type
                for (auto gaussian_idx = start; gaussian_idx < end; ++gaussian_idx) {
                    const auto& gaussian = (*psp_config)[gaussian_idx];
                    kernel_params_t kernel_params;
                    const double gamma = alpha + gaussian.beta;
                    const double lambda = gaussian.beta / gamma;
                    kernel_params.lambda = lambda;
                    kernel_params.gamma = gamma;

                    if (gmp::math::isZero(alpha)) {
                        kernel_params.C1 = gaussian.B;
                        kernel_params.C2 = -1.0 * gaussian.beta;
                    } else {
                        const double temp = std::sqrt(M_PI / gamma);
                        kernel_params.C1 = A * gaussian.B * temp * temp * temp;
                        kernel_params.C2 = -1.0 * (alpha * gaussian.beta / gamma);
                    }
                    table_.emplace_back(kernel_params);
                }
            }
        }
    }

    void kernel_params_table_t::dump() const {
        for (auto i = 0; i < num_features_; ++i) {
            for (auto j = 0; j < num_gaussians_; ++j) {
                std::cout << "kernel_params[" << i << "][" << j << "]: " 
                    << table_[i * num_gaussians_ + j].lambda << ", " 
                    << table_[i * num_gaussians_ + j].gamma << ", " 
                    << table_[i * num_gaussians_ + j].C1 << ", " 
                    << table_[i * num_gaussians_ + j].C2 << std::endl;
            }
        }
    }

    // cutoff_list_t ctor
    cutoff_list_t::cutoff_list_t(const descriptor_config_t* descriptor_config, const psp_config_t* psp_config)
        : cutoff_list_(), cutoff_info_(), gaussian_offset_(psp_config->get_offset()), cutoff_max_(0)
    {
        const auto num_atom_types_ = gaussian_offset_.size() - 1;
        const auto &feature_list = descriptor_config->get_feature_list();
        const auto num_features_ = feature_list.size();
        cutoff_list_.reserve(gaussian_offset_.back() * num_features_);
        cutoff_info_.reserve(gaussian_offset_.back() * num_features_);

        vector<double> factors = {1, 1, 1, 1, 1e-1, 1e-2, 1e-4, 1e-5, 1e-7, 1e-9, 1e-11}; // for threshold 

        for (auto feature_idx = 0; feature_idx < num_features_; ++feature_idx) {
            const auto &feature = feature_list[feature_idx];
            double threshold = descriptor_config->get_overlap_threshold() * factors[feature.order + 1];

            for (auto atom_type_idx = 0; atom_type_idx < num_atom_types_; ++atom_type_idx) {
                auto start = gaussian_offset_[atom_type_idx];
                auto end = gaussian_offset_[atom_type_idx + 1];
                vector<double> cutoff_list;
                for (auto gaussian_idx = start; gaussian_idx < end; ++gaussian_idx) {
                    const auto &gaussian = (*psp_config)[gaussian_idx];
                    const auto cutoff = get_cutoff_gaussian(feature.sigma, gaussian, threshold);
                    cutoff_max_ = std::max(cutoff_max_, cutoff);
                    cutoff_list.push_back(cutoff);
                }
                auto sorted_idx = util::sort_indexes_desc<double, int, vector>(cutoff_list, 0, cutoff_list.size(), start);
                cutoff_info_.insert(cutoff_info_.end(), sorted_idx.begin(), sorted_idx.end());
                for (auto idx : sorted_idx) {
                    cutoff_list_.push_back(cutoff_list[idx - start]);
                }
            }
        }
    }

    void cutoff_list_t::get_range(const int feature_idx, const int atom_idx, int &start, int &end) const {
        const auto shift = feature_idx * gaussian_offset_.back();
        start = shift + gaussian_offset_[atom_idx];
        end = shift + gaussian_offset_[atom_idx + 1];
    }

    // functions 
    double get_scaling_const(const int order, const double sigma, const scaling_mode_t scaling_mode)
    {
        switch (scaling_mode) {
            case scaling_mode_t::radial:
                return get_scaling_constant_radial(sigma);
            case scaling_mode_t::both:
                return get_scaling_constant_both_probes(order, sigma);
            default:
                update_error(gmp::error_t::invalid_scaling_mode);
                return 0.0;
        }
    }

    double get_scaling_constant_radial(const double sigma) 
    {
        const double temp = 1. / (sigma * std::sqrt(2. * M_PI));
        return temp * temp * temp;
    }

    double get_scaling_constant_both_probes(const int order, const double sigma) 
    {
        const double sigma2 = sigma * sigma;
        double temp = pow(M_PI * sigma2, 1.5);
        switch (order) {
        case 0:
            return 1./std::sqrt(temp);
            break;
        
        case 1:
            temp = 0.75 * (2. * sigma2) * temp;
            return 1./std::sqrt(temp);
            break;
        
        case 2:
            temp = 5.625 * pow((2. * sigma2), 2) * temp;
            return 1./std::sqrt(temp);
            break;
        
        case 3:
            temp = 147.65625 * pow((2. * sigma2), 3) * temp;
            return 1./std::sqrt(temp);
            break;
        
        case 4:
            temp = 9302.34375 * pow((2. * sigma2), 4) * temp;
            return 1./std::sqrt(temp);
            break;
        
        case 5:
            temp = 1151165.0390625 * pow((2. * sigma2), 5) * temp;
            return 1./std::sqrt(temp);
            break;
        
        case 6:
            temp = 246924900.87890625 * pow((2. * sigma2), 6) * temp;
            return 1./std::sqrt(temp);
            break;
        
        case 7:
            temp = 84263122424.92676 * pow((2. * sigma2), 7) * temp;
            return 1./std::sqrt(temp);
            break;

        case 8:
            temp = 42974192436712.65 * pow((2. * sigma2), 8) * temp;
            return 1./std::sqrt(temp);
            break;

        case 9:
            temp = 3.163831150894136e16 * pow((2. * sigma2), 9) * temp;
            return 1./std::sqrt(temp);
            break;

        default:
            update_error(gmp::error_t::invalid_order_sigma);
            return 0.0;
        }
    }

    void get_c1_c2(const double A, const double B, const double alpha, const double beta, double& C1, double& C2) 
    {
        if (gmp::math::isZero(alpha)) {
            C1 = B;
            C2 = -1.0 * beta;
        } else {
            double temp = std::sqrt(M_PI / (alpha + beta));
            C1 = A * B * temp * temp * temp;
            C2 = -1.0 * (alpha * beta / (alpha + beta));
        }
    }

    double get_cutoff_gaussian(const double sigma, const input::gaussian_t gaussian, double threshold) 
    {
        double A, alpha;
        if (gmp::math::isZero(sigma)) {
            A = alpha = 0;
        } else {
            A = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
            alpha = 1.0 / (2.0 * sigma * sigma);
        }
        double logThreshold = std::log(threshold);
        
        double C1, C2;
        get_c1_c2(A, gaussian.B, alpha, gaussian.beta, C1, C2);
        C1 = std::abs(C1);
        double dist = std::sqrt((logThreshold - C1) / C2);
        return dist;
    }

    vector<vector<double>> featurizer_t::compute(const vector<point_flt64>& ref_positions, 
        const descriptor_config_t* descriptor_config, const unit_cell_t* unit_cell, const psp_config_t* psp_config)
    {
        // Get the singleton instance first
        const auto& registry = mcsh_function_registry_t::get_instance();
        const auto& region_query = region_query_;
        const auto& feature_list = descriptor_config->get_feature_list();

        vector<vector<double>> feature_collection;
        feature_collection.resize(ref_positions.size());
        
        // Get thread pool for parallel processing
        auto& thread_pool = gmp::resources::gmp_resource::instance().get_thread_pool(descriptor_config->get_num_threads());
        
        // Create futures to store results
        std::vector<std::future<vector<double>>> futures;
        futures.reserve(ref_positions.size());
        
        // Submit tasks to thread pool
        for (size_t i = 0; i < ref_positions.size(); ++i) {
            futures.emplace_back(thread_pool.enqueue([&, i]() -> vector<double> {
                const auto& ref_position = ref_positions[i];
                
                // find neighbors
                auto query_results = region_query->query(ref_position, cutoff_list2_->cutoff_max_, unit_cell);

                // calculate GMP features
                vector<double> feature(feature_list.size(), 0.0);
                for (auto feature_idx = 0; feature_idx < feature_list.size(); ++feature_idx) {
                    auto order = feature_list[feature_idx].order;
                    auto sigma = feature_list[feature_idx].sigma;

                    // Get the function for the specified order
                    const auto& mcsh_func = registry.get_function(order);
                    const auto& num_values = registry.get_num_values(order);
                    vector<double> desc_values(num_values, 0.0);

                    for (const auto& result : query_results) {
                        const auto distance2 = result.distance_squared;
                        const auto& atom = unit_cell->get_atoms()[result.neighbor_index];
                        const auto occ = atom.occ();

                        int start, end;
                        cutoff_list2_->get_range(feature_idx, atom.id(), start, end);

                        for (auto idx = start; idx < end; ++idx) {
                            const auto gaussian_cutoff = cutoff_list2_->cutoff_list_[idx];
                            if (distance2 > gaussian_cutoff * gaussian_cutoff) break;

                            const auto gaussian_idx = cutoff_list2_->cutoff_info_[idx];
                            const auto gaussian = (*psp_config)[gaussian_idx];
                            const auto B = gaussian.B;
                            if (gmp::math::isZero(B)) continue;

                            const auto kernel_params = (*kernel_params_table_)(feature_idx, gaussian_idx);
                            const auto lambda = kernel_params.lambda;
                            const auto gamma = kernel_params.gamma;
                            const auto C1 = kernel_params.C1;
                            const auto C2 = kernel_params.C2;

                            const auto temp = C1 * std::exp(C2 * distance2) * occ;
                            const auto shift = result.difference;

                            mcsh_func(shift, distance2, temp, lambda, gamma, desc_values);
                        }
                    }

                    // weighted square sum of the values
                    double squareSum = weighted_square_sum(order, desc_values);
                    if (descriptor_config->get_square()) {
                        feature[feature_idx] = squareSum;
                    } else {
                        feature[feature_idx] = std::sqrt(squareSum);
                    }
                }
                
                return feature;
            }));
        }
        
        // Collect results
        for (size_t i = 0; i < futures.size(); ++i) {
            feature_collection[i] = futures[i].get();
        }
        
        return feature_collection;
    }
}}

