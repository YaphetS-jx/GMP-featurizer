#pragma once
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include "containers.hpp"
#include "input.hpp"
#include "cuda_region_query.hpp"
#include "atom.hpp"
#include "mcsh.hpp"
#include "gmp_float.hpp"
#include "common_types.hpp"
#include "cuda_tree.hpp"
#include "resources.hpp"
#include "featurizer.hpp"

namespace gmp { namespace featurizer {
    
    using namespace gmp::containers;
    using namespace gmp::input;    
    using namespace gmp::math;
    using namespace gmp::region_query;    
    using namespace gmp::tree;
    using gmp::atom::gaussian_t;

    template <typename T>
    struct cuda_psp_config_t {
        const gaussian_t<T>* __restrict__ table;

        // Device-side accessor
        __device__ __forceinline__
        gaussian_t<T> load(const int idx) const {
            gaussian_t<T> g;
            g.B    = __ldg(&table[idx].B);
            g.beta = __ldg(&table[idx].beta);
            return g; // fields end up in registers
        }
    };

    template <typename T>
    struct cuda_kernel_params_table_t {
        const kernel_params_t<T>* __restrict__ table;
        int num_gaussians;

        // Device-side accessor
        __device__ __forceinline__
        kernel_params_t<T> load(const int feature_idx, const int gaussian_idx) const {
            const int idx = feature_idx * num_gaussians + gaussian_idx;
            return kernel_params_t<T>{
                __ldg(&table[idx].C1),
                __ldg(&table[idx].C2),
                __ldg(&table[idx].lambda),
                __ldg(&table[idx].gamma)
            };
        }
    };

    template <typename T>
    struct cuda_cutoff_list_t {
        const T* __restrict__ cutoff_list;
        const int* __restrict__ cutoff_info;
        const int* __restrict__ gaussian_offset;
        int num_gaussians;
        T cutoff_max;

        // Device-side accessor - returns range for feature_idx, atom_idx
        __device__ __forceinline__
        void get_range(const int feature_idx, const int atom_idx, int &start, int &end) const {
            const auto shift = feature_idx * num_gaussians;
            start = shift + gaussian_offset[atom_idx];
            end = shift + gaussian_offset[atom_idx + 1];
        }

        __device__ __forceinline__
        int load_offset(const int idx) const {
            return __ldg(&gaussian_offset[idx]);
        }

        __device__ __forceinline__
        int load_info(const int feature_idx, const int idx) const {
            return __ldg(&cutoff_info[idx]);
        }

        __device__ __forceinline__
        T load_cutoff(const int feature_idx, const int idx) const {
            const auto shift = feature_idx * num_gaussians;
            return __ldg(&cutoff_list[shift + idx]);
        }
    };

    // Main CUDA featurizer class
    template <typename T>
    class cuda_featurizer_t {
    public:
        cuda_featurizer_t(
            // query points 
            const vector<point3d_t<T>>& h_positions,
            // host atoms 
            const vector<atom_t<T>>& h_atoms,
            // host psp_config 
            const psp_config_t<T>* h_psp_config,
            // host kernel_params_table
            const kernel_params_table_t<T>* h_kernel_params_table,
            // host cutoff_list
            const cutoff_list_t<T>* h_cutoff_list,
            // region query initialization parameters
            const vector<uint32_t>& h_morton_codes, const int32_t num_bits_per_dim, 
            const vector<int32_t>& h_offsets, const vector<int32_t>& h_sorted_indexes, 
            // stream
            cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream());
        
    public:
        // device data
        vector_device<point3d_t<T>> d_positions;
        vector_device<atom_t<T>> d_atoms;
        // data for cuda_psp_config_t
        vector_device<gaussian_t<T>> d_gaussian_table;
        // data for cuda_kernel_params_table_t
        vector_device<kernel_params_t<T>> d_kernel_params_table;
        // data for cuda_cutoff_list_t
        vector_device<T> d_cutoff_list;
        vector_device<int> d_cutoff_info;
        vector_device<int> d_cutoff_gaussian_offset;

        // device region query
        std::unique_ptr<cuda_region_query_t<uint32_t, int32_t, T>> d_region_query;

        // device POD structures
        cuda_psp_config_t<T> d_psp_config;
        cuda_kernel_params_table_t<T> d_kernel_params;
        cuda_cutoff_list_t<T> d_cutoff;
    
    public: 
    vector<T> compute(
            std::vector<feature_t<T>> feature_list, const bool feature_square, const lattice_t<T>* lattice, 
            cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream());
    };

    // Type aliases using configured floating-point type
    using cuda_kernel_params_table_flt = cuda_kernel_params_table_t<gmp::gmp_float>;
    using cuda_cutoff_list_flt = cuda_cutoff_list_t<gmp::gmp_float>;
    using cuda_featurizer_flt = cuda_featurizer_t<gmp::gmp_float>;
}}
