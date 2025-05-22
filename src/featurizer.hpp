#pragma once
#include "types.hpp"
#include "input.hpp"
#include "query.hpp"

namespace gmp { namespace featurizer {
    
    using namespace gmp::containers;
    using namespace gmp::input;
    using namespace gmp::query;
    using namespace gmp::math;

    // basic structures
    struct kernel_params_t {
        double C1, C2;
        double lambda, gamma;
    };

    class kernel_params_table_t {
    public:
        kernel_params_table_t(const vec<int>&& gaussian_offset, const int num_gaussians, const int num_features) 
            : gaussian_offset_(std::move(gaussian_offset)), num_gaussians_(num_gaussians), num_features_(num_features) {};
        ~kernel_params_table_t() = default;

        void set_kernel_params_table(const descriptor_config_t& descriptor_config, const psp_config_t& psp_config);

        const kernel_params_t& operator()(const int feature_idx, const int atom_type_idx, const int gaussian_idx) const {
            return table_[feature_idx * num_gaussians_ + get_gaussian(atom_type_idx, gaussian_idx)];
        }

        int get_gaussian(const int atom_type_idx, const int gaussian_idx) const {
            return gaussian_offset_[atom_type_idx] + gaussian_idx;
        }

    private: 
        vec<kernel_params_t> table_;
        vec<int> gaussian_offset_;
        int num_gaussians_;
        int num_features_;
    };

    class cutoff_table_t {
    public:
        cutoff_table_t() 
            : feature_mapping_(), gaussian_mapping_(), largest_cutoff_(0.0), feature_size_(0), gaussian_size_(0) {};
        ~cutoff_table_t() = default;

        int get_feature_idx(const int feature_idx) const {
            return feature_mapping_[feature_idx];
        }

        int get_gaussian_idx(const int gaussian_idx) const {
            return gaussian_mapping_[gaussian_idx];
        }

        double operator()(const int feature_idx, const int gaussian_idx) const {
            return table_[get_feature_idx(feature_idx) * gaussian_size_ + get_gaussian_idx(gaussian_idx)];
        }

        void set_cutoff_table(const gmp::input::descriptor_config_t& descriptor_config, const gmp::input::psp_config_t& psp_config);


        double get_largest_cutoff() const { return largest_cutoff_; }

        double get_cufoff_table_2(const int feature_idx, const int element_idx) const {
            return table2_[get_feature_idx(feature_idx) * gaussian_mapping_.size() + element_idx];
        }

    private: 
        vec<double> table_;
        vec<double> table2_;
        vec<int> feature_mapping_;  // outer index
        vec<int> gaussian_mapping_; // inner index
        double largest_cutoff_;
        int feature_size_;
        int gaussian_size_;
    };


    // featurizer class
    class featurizer_t {
    public:
        featurizer_t() : query_info_(), kernel_params_table_(), cutoff_table_() {};
        ~featurizer_t() = default;

        // calculate features 
        vec<vec<double>> operator()(const vec<point_flt64>& ref_positions, 
            const descriptor_config_t& descriptor_config, const unit_cell_t& unit_cell, const psp_config_t& psp_config);

    private: 
        gmp_unique_ptr<query_info_t> query_info_;
        gmp_unique_ptr<kernel_params_table_t> kernel_params_table_;
        gmp_unique_ptr<cutoff_table_t> cutoff_table_;
    };

    // functions 
    double get_scaling_const(const int order, const double sigma, const gmp::input::scaling_mode_t scaling_mode);
    double get_scaling_constant_radial(const double sigma);
    double get_scaling_constant_both_probes(const int order, const double sigma);
    void get_c1_c2(double A, double B, double alpha, double beta, double& C1, double& C2);
    double get_cutoff_gaussian(const double sigma, const input::gaussian_t gaussian, double threshold);
}}