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
        kernel_params_t() : C1(0.0), C2(0.0), lambda(0.0), gamma(0.0) {}
        kernel_params_t(const double C1, const double C2, const double lambda, const double gamma) : C1(C1), C2(C2), lambda(lambda), gamma(gamma) {}
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
        vector<kernel_params_t> table_;
        int num_gaussians_;
        int num_features_;
    };

    class cutoff_list_t {
    public:
        cutoff_list_t(const descriptor_config_t* descriptor_config, const psp_config_t* psp_config);
        ~cutoff_list_t() = default;

        void get_range(const int feature_idx, const int atom_idx, int &start, int &end) const;
    public: 
        vector<double> cutoff_list_;
        vector<int> cutoff_info_;
        vector<int> gaussian_offset_;
        double cutoff_max_;
    };


    // featurizer class
    class featurizer_t {
    public:
        using query_t = region_query_t<uint32_t, int32_t, double, vector<array3d_int32>>;
        // ctor
        featurizer_t(const descriptor_config_t* descriptor_config, const unit_cell_t* unit_cell, const psp_config_t* psp_config)
            : kernel_params_table_(std::make_unique<kernel_params_table_t>(descriptor_config, psp_config)), 
            cutoff_list2_(std::make_unique<cutoff_list_t>(descriptor_config, psp_config)),
            region_query_(std::make_unique<query_t>(unit_cell, 4))
        {}

        ~featurizer_t() = default;

        // calculate features 
        vector<vector<double>> compute(const vector<point_flt64>& ref_positions, 
            const descriptor_config_t* descriptor_config, const unit_cell_t* unit_cell, const psp_config_t* psp_config);

        // vector<vector<double>> compute_simd(const vector<point_flt64>& ref_positions, 
        //     const descriptor_config_t* descriptor_config, const unit_cell_t* unit_cell, const psp_config_t* psp_config);
        
    private: 
        std::unique_ptr<kernel_params_table_t> kernel_params_table_;
        std::unique_ptr<cutoff_list_t> cutoff_list2_;
        std::unique_ptr<query_t> region_query_;
    };

    // functions 
    double get_scaling_const(const int order, const double sigma, const gmp::input::scaling_mode_t scaling_mode);
    double get_scaling_constant_radial(const double sigma);
    double get_scaling_constant_both_probes(const int order, const double sigma);
    void get_c1_c2(double A, double B, double alpha, double beta, double& C1, double& C2);
    double get_cutoff_gaussian(const double sigma, const input::gaussian_t gaussian, double threshold);
}}