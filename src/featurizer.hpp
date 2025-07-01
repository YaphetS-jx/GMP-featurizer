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
    template <typename T>
    struct kernel_params_t {
        T C1, C2;
        T lambda, gamma;
        kernel_params_t() : C1(0.0), C2(0.0), lambda(0.0), gamma(0.0) {}
        kernel_params_t(const T C1, const T C2, const T lambda, const T gamma) : C1(C1), C2(C2), lambda(lambda), gamma(gamma) {}
    };

    template <typename T>
    class kernel_params_table_t {
    public:
        kernel_params_table_t(const descriptor_config_t<T>* descriptor_config, const psp_config_t<T>* psp_config);
        ~kernel_params_table_t() = default;

        const kernel_params_t<T>& operator()(const int feature_idx, const int gaussian_idx) const {
            return table_[feature_idx * num_gaussians_ + gaussian_idx];
        }

        void dump() const;

    private: 
        vector<kernel_params_t<T>> table_;
        int num_gaussians_;
        int num_features_;
    };

    template <typename T>
    class cutoff_list_t {
    public:
        cutoff_list_t(const descriptor_config_t<T>* descriptor_config, const psp_config_t<T>* psp_config);
        ~cutoff_list_t() = default;

        void get_range(const int feature_idx, const int atom_idx, int &start, int &end) const;
    public: 
        vector<T> cutoff_list_;
        vector<int> cutoff_info_;
        vector<int> gaussian_offset_;
        T cutoff_max_;
    };


    // featurizer class
    template <typename T>
    class featurizer_t {
    public:
        using query_t = region_query_t<uint32_t, int32_t, T, vector<array3d_int32>>;
        // ctor
        featurizer_t(const descriptor_config_t<T>* descriptor_config, const unit_cell_t<T>* unit_cell, const psp_config_t<T>* psp_config)
            : kernel_params_table_(std::make_unique<kernel_params_table_t<T>>(descriptor_config, psp_config)), 
            cutoff_list_(std::make_unique<cutoff_list_t<T>>(descriptor_config, psp_config)),
            region_query_(std::make_unique<query_t>(unit_cell, descriptor_config->get_num_bits_per_dim()))
        {}

        ~featurizer_t() = default;

        // calculate features 
        vector<vector<T>> compute(const vector<point3d_t<T>>& ref_positions, 
            const descriptor_config_t<T>* descriptor_config, const unit_cell_t<T>* unit_cell, const psp_config_t<T>* psp_config);

        // vector<vector<T>> compute_simd(const vector<point3d_t<T>>& ref_positions, 
        //     const descriptor_config_t* descriptor_config, const unit_cell_t* unit_cell, const psp_config_t* psp_config);
        
    private: 
        std::unique_ptr<kernel_params_table_t<T>> kernel_params_table_;
        std::unique_ptr<cutoff_list_t<T>> cutoff_list_;
        std::unique_ptr<query_t> region_query_;
    };

    // functions 
    template <typename T>
    T get_scaling_const(const int order, const T sigma, const gmp::input::scaling_mode_t scaling_mode);
    
    template <typename T>
    T get_scaling_constant_radial(const T sigma);
    
    template <typename T>
    T get_scaling_constant_both_probes(const int order, const T sigma);
    
    template <typename T>
    void get_c1_c2(T A, T B, T alpha, T beta, T& C1, T& C2);
    
    template <typename T>
    T get_cutoff_gaussian(const T sigma, const gaussian_t<T> gaussian, T threshold);

    // Type aliases for common types
    using kernel_params_flt64 = kernel_params_t<double>;
    using kernel_params_table_flt64 = kernel_params_table_t<double>;
    using cutoff_list_flt64 = cutoff_list_t<double>;
    using featurizer_flt64 = featurizer_t<double>;
}}

#include "featurizer.hxx"