#pragma once
#include "types.hpp"
#include "input.hpp"
#include "region_query.hpp"

namespace gmp { namespace featurizer {
    
    using namespace gmp::containers;
    using namespace gmp::input;    
    using namespace gmp::math;
    using namespace gmp::region_query;

    // basic structures
    struct kernel_params_t {
        double C1, C2;
        double lambda, gamma;
    };

    class kernel_params_table_t {
    public:
        kernel_params_table_t(const descriptor_config_t* descriptor_config, const psp_config_t* psp_config);
        ~kernel_params_table_t() = default;

        const kernel_params_t& operator()(const int feature_idx, const int gaussian_idx) const {
            return table_[feature_idx * num_gaussians_ + gaussian_idx];
        }

        void dump() const;

    private: 
        vec<kernel_params_t> table_;
        int num_gaussians_;
        int num_features_;
    };

    class cutoff_table_t {
    public:
        cutoff_table_t(const descriptor_config_t* descriptor_config, const psp_config_t* psp_config);
        ~cutoff_table_t() = default;

        int get_feature_idx(const int feature_idx) const {
            return feature_mapping_[feature_idx];
        }

        int get_gaussian_idx(const int gaussian_idx) const {
            return gaussian_mapping_[gaussian_idx];
        }

        double operator()(const int feature_idx, const int gaussian_idx) const {
            return table_[get_feature_idx(feature_idx) * num_gaussians_ + get_gaussian_idx(gaussian_idx)];
        }

        double get_largest_cutoff() const { return largest_cutoff_; }

        double get_cufoff_table_2(const int feature_idx, const int element_idx) const {
            return table2_[get_feature_idx(feature_idx) * num_atom_types_ + element_idx];
        }

        void dump() const;

    private: 
        vec<double> table_;
        vec<double> table2_;
        vec<int> feature_mapping_;  // outer index
        vec<int> gaussian_mapping_; // inner index
        double largest_cutoff_;
        int num_features_;
        int num_gaussians_;
        int num_atom_types_;
    };


    // featurizer class
    class featurizer_t {
    public:
        using query_t = region_query_t<uint32_t, int32_t, double, vec<array3d_int32>>;
        // ctor
        featurizer_t(const descriptor_config_t* descriptor_config, const unit_cell_t* unit_cell, const psp_config_t* psp_config)
            : kernel_params_table_(std::make_unique<kernel_params_table_t>(descriptor_config, psp_config)), 
            cutoff_table_(std::make_unique<cutoff_table_t>(descriptor_config, psp_config)),            
            region_query_(std::make_unique<query_t>(unit_cell, 4))
        {}
        ~featurizer_t() = default;

        // calculate features 
        vec<vec<double>> compute(const vec<point_flt64>& ref_positions, 
            const descriptor_config_t* descriptor_config, const unit_cell_t* unit_cell, const psp_config_t* psp_config);

    private: 
        std::unique_ptr<kernel_params_table_t> kernel_params_table_;
        std::unique_ptr<cutoff_table_t> cutoff_table_;        
        std::unique_ptr<query_t> region_query_;
    };

    // functions 
    double get_scaling_const(const int order, const double sigma, const gmp::input::scaling_mode_t scaling_mode);
    double get_scaling_constant_radial(const double sigma);
    double get_scaling_constant_both_probes(const int order, const double sigma);
    void get_c1_c2(double A, double B, double alpha, double beta, double& C1, double& C2);
    double get_cutoff_gaussian(const double sigma, const input::gaussian_t gaussian, double threshold);
}}