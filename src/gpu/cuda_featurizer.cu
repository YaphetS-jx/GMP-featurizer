#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "error.hpp"
#include "math.hpp"
#include "input.hpp"
#include "containers.hpp"
#include "util.hpp"
#include "region_query.hpp"
#include "mcsh.hpp"
#include "resources.hpp"
#include "cuda_featurizer.hpp"
#include "cuda_thrust_ops.hpp"
#include "cuda_util.hpp"
#include "gmp_float.hpp"

namespace gmp { namespace featurizer {    

    using namespace gmp::thrust_ops;
    using namespace gmp::group_add;

    // Main CUDA featurizer implementation
    template <typename T>
    cuda_featurizer_t<T>::cuda_featurizer_t(
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
        cudaStream_t stream)
        : d_positions(h_positions.size(), stream),
        d_atoms(h_atoms.size(), stream),
        d_gaussian_table(h_psp_config->gaussian_table_.size(), stream),
        d_kernel_params_table(h_kernel_params_table->table_.size(), stream),
        d_cutoff_list(h_cutoff_list->cutoff_list_.size(), stream),
        d_cutoff_info(h_cutoff_list->cutoff_info_.size(), stream),
        d_cutoff_gaussian_offset(h_cutoff_list->gaussian_offset_.size(), stream),
        d_region_query(std::make_unique<cuda_region_query_t<uint32_t, int32_t, T>>(
            h_morton_codes, num_bits_per_dim, h_offsets, h_sorted_indexes, stream))
    {
        // Copy data to device 
        cudaMemcpyAsync(d_positions.data(), h_positions.data(), h_positions.size() * sizeof(point3d_t<T>), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_atoms.data(), h_atoms.data(), h_atoms.size() * sizeof(atom_t<T>), cudaMemcpyHostToDevice, stream);

        // save device memory for cuda_psp_config_t, cuda_kernel_params_table_t, cuda_cutoff_list_t
        cudaMemcpyAsync(d_gaussian_table.data(), h_psp_config->gaussian_table_.data(), d_gaussian_table.size() * sizeof(gaussian_t<T>), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_kernel_params_table.data(), h_kernel_params_table->table_.data(), d_kernel_params_table.size() * sizeof(kernel_params_t<T>), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_cutoff_list.data(), h_cutoff_list->cutoff_list_.data(), d_cutoff_list.size() * sizeof(T), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_cutoff_info.data(), h_cutoff_list->cutoff_info_.data(), d_cutoff_info.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_cutoff_gaussian_offset.data(), h_cutoff_list->gaussian_offset_.data(), d_cutoff_gaussian_offset.size() * sizeof(int), cudaMemcpyHostToDevice, stream);

        cudaStreamSynchronize(stream);

        // create device POD structures 
        d_psp_config = cuda_psp_config_t<T>{d_gaussian_table.data()};
        d_kernel_params = cuda_kernel_params_table_t<T>{d_kernel_params_table.data(), h_kernel_params_table->num_gaussians_};
        d_cutoff = cuda_cutoff_list_t<T>{d_cutoff_list.data(), d_cutoff_info.data(), d_cutoff_gaussian_offset.data(), 
            h_cutoff_list->gaussian_offset_.back(), h_cutoff_list->cutoff_max_};
    }

    template <typename T>
    __global__
    void get_num_gaussian_list(
        const cuda_query_result_t<T>* query_results, const int num_query_results, 
        const atom_t<T>* atoms,
        const cuda_cutoff_list_t<T> cutoff_list, 
        int* num_gaussian_list)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_query_results) return;

        auto atom_idx = query_results[idx].neighbor_index;
        auto atom = atoms[atom_idx];
        auto start = cutoff_list.load_offset(atom.type_id);
        auto end = cutoff_list.load_offset(atom.type_id + 1);

        num_gaussian_list[idx] = end - start;
    }

    template <typename T>
    __global__
    void get_target_list_kernel(const int feature_idx,
        const int* num_gaussian_offset, const int num_query_results, 
        const int* query_idx_mapping, const int total_num_gaussians, 
        const cuda_query_result_t<T>* query_results, 
        const atom_t<T>* atoms,
        const cuda_cutoff_list_t<T> cutoff_list, 
        int32_t* target_list)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_num_gaussians) return;

        auto query_idx = query_idx_mapping[idx];
        assert(query_idx < num_query_results);
        auto result = query_results[query_idx];
        auto distance2 = result.distance_squared;
        
        auto local_gaussian_idx = query_idx > 0 ? (idx - num_gaussian_offset[query_idx - 1]) : idx;
        auto atom = atoms[result.neighbor_index];
        auto gaussian_cutoff = cutoff_list.load_cutoff(feature_idx, atom.type_id, local_gaussian_idx);
        target_list[idx] = (distance2 > gaussian_cutoff * gaussian_cutoff) ? 0 : 1;
    }

    template <typename T> 
    vector_device<int32_t> get_target_list(
        const int feature_idx, const vector_device<int>& num_gaussian_offset, const int num_query_results, 
        const vector_device<int>& query_idx_mapping, const int total_num_gaussians, 
        const vector_device<cuda_query_result_t<T>>& query_results, 
        const vector_device<atom_t<T>>& atoms,
        const cuda_cutoff_list_t<T> cutoff_list, 
        gmp::resources::device_memory_manager* dm, cudaStream_t stream)
    {
        dim3 block_size(64, 1, 1), grid_size(1, 1, 1);
        // template memory for calculation index 
        vector_device<int32_t> temp_target_list(total_num_gaussians, stream);

        // filtering out the target list 
        grid_size.x = (total_num_gaussians + block_size.x - 1) / block_size.x;
        get_target_list_kernel<<<grid_size, block_size, 0, stream>>>(
            feature_idx, num_gaussian_offset.data(), num_query_results, query_idx_mapping.data(), 
            total_num_gaussians, query_results.data(), atoms.data(), cutoff_list, temp_target_list.data());

        // compact target list 
        vector_device<int32_t> target_list(total_num_gaussians, stream);
        auto num_selected = compact(temp_target_list, target_list, dm, stream);
        target_list.resize(num_selected, stream);
        return target_list;
    }

    __global__
    void get_query_indexes_kernel(
        const int32_t* target_list, const int num_selected,
        const int* query_idx_mapping, int32_t* query_indexes)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_selected) return;

        int32_t target_idx = target_list[idx];
        query_indexes[idx] = query_idx_mapping[target_idx];
    }

    vector_device<int32_t> get_query_indexes(
        const vector_device<int32_t>& target_list,
        const vector_device<int>& query_idx_mapping, 
        gmp::resources::device_memory_manager* dm, cudaStream_t stream)
    {
        dim3 block_size(64, 1, 1), grid_size(1, 1, 1);
        // template memory for calculation index 
        auto num_selected = target_list.size();
        vector_device<int32_t> query_indexes(num_selected, stream);

        // filtering out the target list 
        grid_size.x = (num_selected + block_size.x - 1) / block_size.x;
        get_query_indexes_kernel<<<grid_size, block_size, 0, stream>>>(
            target_list.data(), num_selected, query_idx_mapping.data(), query_indexes.data());
        return query_indexes;
    }

    template <int Order, typename T> 
    __global__
    void featurizer_kernel(const int feature_idx,
        const int32_t* target_list, const int num_selected, 
        const int32_t* num_gaussian_offset, 
        const int32_t* query_indexes, const int32_t* point_indexes,
        const cuda_query_result_t<T>* query_results, const int num_query_results, 
        const atom_t<T>* atoms, 
        const cuda_psp_config_t<T> psp_config, 
        const cuda_cutoff_list_t<T> cutoff_list, 
        const cuda_kernel_params_table_t<T> kernel_params_table, 
        T* desc_values, const int num_values
    )
    {
        // Shared memory hash table
        constexpr int HASH_SIZE = 128;  // Increase size to reduce collisions
        __shared__ int   hash_keys[HASH_SIZE];
        __shared__ T hash_vals[HASH_SIZE];
        
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int local_tid = threadIdx.x;
        
        // Create hash table structure
        HashTable<T> hash_table(hash_keys, hash_vals, HASH_SIZE);
        
        // Initialize hash table
        hash_table.initialize(local_tid, blockDim.x);
        __syncthreads();

        // Process elements
        if (tid < num_selected) {
            int32_t target_idx = target_list[tid];
            auto query_idx = query_indexes[tid];
            assert(query_idx < num_query_results);
            auto point_idx = point_indexes[tid];

            // get gaussian 
            auto local_gaussian_idx = query_idx > 0 ? (target_idx - num_gaussian_offset[query_idx - 1]) : target_idx;
            auto result = query_results[query_idx];
            auto atom = atoms[result.neighbor_index];
            auto gaussian_idx = cutoff_list.load_info(feature_idx, atom.type_id, local_gaussian_idx);
            auto gaussian = psp_config.load(gaussian_idx);
            auto B = gaussian.B;
            // if (gmp::math::isZero(B)) continue;

            // get kernel parameters
            auto kp = kernel_params_table.load(feature_idx, gaussian_idx);

            // get query result
            auto occ = atom.occ;
            const auto temp = kp.C1 * std::exp(kp.C2 * result.distance_squared) * occ;
            const auto shift = result.difference;

            // calculate mcsh values
            mcsh::solid_mcsh<Order>(shift, result.distance_squared, temp, kp.lambda, kp.gamma, 
                desc_values, point_idx * num_values, &hash_table, local_tid);
        }

        __syncthreads();
        // Flush hash table to global memory
        for (int i = local_tid; i < HASH_SIZE; i += blockDim.x) {
            if (hash_keys[i] != -1 && hash_vals[i] != 0.0) {
                atomicAdd(desc_values + hash_keys[i], hash_vals[i]);
            }
        }
    }

    template <typename T>
    void launch_featurizer_kernel(
        const int order, dim3 grid_size, dim3 block_size, cudaStream_t stream,
        const int feature_idx, const int32_t* target_list, const int num_selected,
        const int32_t* num_gaussian_offset,
        const int32_t* query_indexes, const int32_t* point_indexes,
        const cuda_query_result_t<T>* query_results, const int num_query_results,
        const atom_t<T>* atoms,
        const cuda_psp_config_t<T> psp_config,
        const cuda_cutoff_list_t<T> cutoff_list,
        const cuda_kernel_params_table_t<T> kernel_params_table,
        T* desc_values, const int num_values)
    {
        switch (order) {
#define LAUNCH_FEATURIZER_CASE(order_value) \
            case order_value: \
                featurizer_kernel<order_value, T><<<grid_size, block_size, 0, stream>>>( \
                    feature_idx, target_list, num_selected, \
                    num_gaussian_offset, \
                    query_indexes, point_indexes, \
                    query_results, num_query_results, \
                    atoms, \
                    psp_config, \
                    cutoff_list, \
                    kernel_params_table, \
                    desc_values, num_values); \
                break;
            LAUNCH_FEATURIZER_CASE(-1)
            LAUNCH_FEATURIZER_CASE(0)
            LAUNCH_FEATURIZER_CASE(1)
            LAUNCH_FEATURIZER_CASE(2)
            LAUNCH_FEATURIZER_CASE(3)
            LAUNCH_FEATURIZER_CASE(4)
            LAUNCH_FEATURIZER_CASE(5)
            LAUNCH_FEATURIZER_CASE(6)
            LAUNCH_FEATURIZER_CASE(7)
            LAUNCH_FEATURIZER_CASE(8)
            LAUNCH_FEATURIZER_CASE(9)
#undef LAUNCH_FEATURIZER_CASE
            default:
                throw std::invalid_argument("Unsupported MCSH order");
        }
    }


    template <typename T>
    __global__
    void weighted_square_sum_kernel(const int order, const int num_values, const int num_positions,
        const bool square, const T* desc_values, T* feature)
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_positions) return;

        T squareSum = gmp::mcsh::weighted_square_sum(order, desc_values + idx * num_values);
        if (square) {
            feature[idx] = squareSum;
        } else {
            feature[idx] = std::sqrt(squareSum);
        }
    }

    template <typename T>
    __global__
    void copy_raw_data_kernel(const int num_values, const int num_positions, const int total_feature_size,
        const int raw_offset, const T* desc_values, T* feature_collection)
    {
        auto pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pos_idx >= num_positions) return;
        
        const T* src = desc_values + pos_idx * num_values;
        T* dst = feature_collection + pos_idx * total_feature_size + raw_offset;
        
        for (int i = 0; i < num_values; ++i) {
            dst[i] = src[i];
        }
    }

    template <typename T>
    vector<T> cuda_featurizer_t<T>::compute(
        std::vector<feature_t<T>> feature_list, const bool feature_square, const lattice_t<T>* lattice, 
        cudaStream_t stream, const bool output_raw_data)
    {
        // get device memory manager
        auto dm = gmp::resources::gmp_resource::instance().get_device_memory_manager();

        // kernel parameters
        dim3 block_size(64, 1, 1), grid_size(1, 1, 1);

        // cuda region query 
        vector_device<cuda_query_result_t<T>> query_results(1, stream);
        vector_device<int32_t> query_offsets(1, stream);
        cuda_region_query(d_positions, d_cutoff.cutoff_max, *d_region_query, 
            lattice, d_atoms, query_results, query_offsets, stream);
        
        // find out number of gaussian list and offset
        auto num_query_results = query_results.size();
        vector_device<int32_t> num_gaussian_offset(num_query_results, stream);
        grid_size.x = (num_query_results + block_size.x - 1) / block_size.x;
        get_num_gaussian_list<<<grid_size, block_size, 0, stream>>>(
            query_results.data(), num_query_results, d_atoms.data(), d_cutoff, num_gaussian_offset.data());

        // calculate offset
        THRUST_CALL(thrust::inclusive_scan, dm, stream, 
            num_gaussian_offset.data(), num_gaussian_offset.data() + num_query_results, num_gaussian_offset.data());
        
        int32_t total_num_gaussians;
        cudaMemcpyAsync(&total_num_gaussians, num_gaussian_offset.data() + num_query_results - 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // get index mapping 
        vector_device<int32_t> query_idx_mapping(total_num_gaussians, stream);
        get_index_mapping(num_gaussian_offset, query_idx_mapping, dm, stream);

        // featurizer calculation 
        auto num_positions = d_positions.size();
        auto num_features = feature_list.size();
        
        // Calculate total feature size
        size_t total_feature_size = num_features;
        if (output_raw_data) {
            total_feature_size = 0;
            for (const auto& feature : feature_list) {
                int order = feature.order;
                total_feature_size += (order < 0 ? 1 : mcsh::num_mcsh_values[order]);
            }
        }
        
        vector_device<T> feature_collection(num_positions * total_feature_size, stream);
        cudaMemsetAsync(feature_collection.data(), 0, num_positions * total_feature_size * sizeof(T), stream);

        for (auto feature_idx = 0; feature_idx < num_features; ++feature_idx) {
            auto order = feature_list[feature_idx].order;
            auto sigma = feature_list[feature_idx].sigma;
            auto num_values = order < 0 ? 1 : mcsh::num_mcsh_values[order];

            // get target list 
            auto target_list = get_target_list(feature_idx, 
                num_gaussian_offset, num_query_results, query_idx_mapping, 
                total_num_gaussians, query_results, d_atoms, d_cutoff, dm, stream);
            
            auto num_selected = target_list.size();

            // get query_indexes
            auto query_indexes = get_query_indexes(target_list, query_idx_mapping, dm, stream);

            // get point indexes
            vector_device<int32_t> point_indexes(num_selected, stream);
            get_index_mapping(query_offsets, query_indexes, point_indexes, dm, stream);

            // doing calculation
            vector_device<T> desc_values(num_values * num_positions, stream);
            cudaMemsetAsync(desc_values.data(), 0, num_values * num_positions * sizeof(T), stream);
            grid_size.x = (num_selected + block_size.x - 1) / block_size.x;
            launch_featurizer_kernel<T>(order, grid_size, block_size, stream,
                feature_idx, target_list.data(), num_selected,
                num_gaussian_offset.data(),
                query_indexes.data(), point_indexes.data(),
                query_results.data(), num_query_results,
                d_atoms.data(), d_psp_config, d_cutoff, d_kernel_params,
                desc_values.data(), num_values);
            
            if (output_raw_data) {
                // Copy raw data directly (all values for all positions)
                // Calculate offset in feature_collection for this feature
                size_t raw_offset = 0;
                for (int i = 0; i < feature_idx; ++i) {
                    int feat_order = feature_list[i].order;
                    raw_offset += (feat_order < 0 ? 1 : mcsh::num_mcsh_values[feat_order]);
                }
                
                // Use kernel to copy raw data in parallel for all positions
                grid_size.x = (num_positions + block_size.x - 1) / block_size.x;
                copy_raw_data_kernel<<<grid_size, block_size, 0, stream>>>(
                    num_values, num_positions, total_feature_size,
                    raw_offset, desc_values.data(), feature_collection.data());
            } else {
                // weighted square sum of the values
                grid_size.x = (num_positions + block_size.x - 1) / block_size.x;
                weighted_square_sum_kernel<<<grid_size, block_size, 0, stream>>>(
                    order, num_values, num_positions, 
                    feature_square, desc_values.data(), 
                    feature_collection.data() + feature_idx * num_positions);
            }
        }
        
        // copy to host 
        vector<T> h_feature_collection(feature_collection.size());
        cudaMemcpyAsync(h_feature_collection.data(), feature_collection.data(), feature_collection.size() * sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        return h_feature_collection;
    }

    // Explicit instantiations
    template class cuda_featurizer_t<gmp::gmp_float>;

}} // namespace gmp::featurizer
