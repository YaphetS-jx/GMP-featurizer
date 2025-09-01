#include <algorithm>
#include <cmath>
#include "cuda_region_query.hpp"
#include "cuda_tree.hpp"
#include "cuda_util.hpp"
#include "cuda_thrust_ops.hpp"
#include <thrust/scan.h>

namespace gmp { namespace region_query {

    using namespace gmp::util;
    using gmp::thrust_ops::ThrustAllocator;

    template <typename FloatType, typename IndexType>
    __global__
    void count_cell_shifts_kernel(const array3d_t<FloatType> frac_radius, const point3d_t<FloatType>* positions, const IndexType num_positions, 
        IndexType* offset_cshift)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_positions) return;
        const auto& position = positions[tid];

        IndexType cell_shift_start_x = floor(position.x - frac_radius[0]);
        IndexType cell_shift_end_x = floor(position.x + frac_radius[0]);
        IndexType cell_shift_start_y = floor(position.y - frac_radius[1]);
        IndexType cell_shift_end_y = floor(position.y + frac_radius[1]);
        IndexType cell_shift_start_z = floor(position.z - frac_radius[2]);
        IndexType cell_shift_end_z = floor(position.z + frac_radius[2]);

        offset_cshift[tid] = (cell_shift_end_x - cell_shift_start_x + 1) 
                    * (cell_shift_end_y - cell_shift_start_y + 1) 
                    * (cell_shift_end_z - cell_shift_start_z + 1);
    }

    template <typename FloatType, typename IndexType>
    __global__
    void get_cell_shifts_kernel(const array3d_t<FloatType> frac_radius, const point3d_t<FloatType>* positions, const IndexType num_positions, 
        const IndexType* offset_cshift, IndexType* query_target_indexes, array3d_t<IndexType>* query_target_cell_shifts, const IndexType num_queries)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_queries) return;

        auto pos_index = gmp::util::binary_search_first_larger(offset_cshift, IndexType(0), num_positions - 1, IndexType(tid));
        const auto& position = positions[pos_index];
        auto local_index = pos_index ? (tid - offset_cshift[pos_index - 1]) : tid;

        IndexType cell_shift_start_x = floor(position.x - frac_radius[0]);
        IndexType cell_shift_end_x = floor(position.x + frac_radius[0]);
        IndexType cell_shift_start_y = floor(position.y - frac_radius[1]);
        IndexType cell_shift_end_y = floor(position.y + frac_radius[1]);
        IndexType cell_shift_start_z = floor(position.z - frac_radius[2]);
        IndexType cell_shift_end_z = floor(position.z + frac_radius[2]);
        IndexType cell_shift_x = cell_shift_end_x - cell_shift_start_x + 1;
        IndexType cell_shift_y = cell_shift_end_y - cell_shift_start_y + 1;
        IndexType cell_shift_z = cell_shift_end_z - cell_shift_start_z + 1;

        // get local index for cell shift
        IndexType local_index_x = local_index % cell_shift_x;
        IndexType local_index_y = (local_index / cell_shift_x) % cell_shift_y;
        IndexType local_index_z = local_index / (cell_shift_x * cell_shift_y);

        query_target_indexes[tid] = pos_index;
        query_target_cell_shifts[tid] = {cell_shift_start_x + local_index_x, 
                                         cell_shift_start_y + local_index_y, 
                                         cell_shift_start_z + local_index_z};
    }

    template <typename IndexType, typename FloatType>
    void get_query_targets(const array3d_t<FloatType> frac_radius, const vector_device<point3d_t<FloatType>>& positions, IndexType& num_queries, 
        vector_device<IndexType>& query_target_indexes, vector_device<array3d_t<IndexType>>& query_target_cell_shifts)
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        auto dm = gmp::resources::gmp_resource::instance().get_device_memory_manager();

        auto num_positions = static_cast<uint32_t>(positions.size());
        vector_device<IndexType> offset_cshift(num_positions, stream);

        dim3 block_size(256, 1, 1), grid_size(1, 1, 1);
        grid_size.x = (num_positions + block_size.x - 1) / block_size.x;
        count_cell_shifts_kernel<FloatType, IndexType><<<grid_size, block_size, 0, stream>>>(frac_radius, positions.data(), num_positions, offset_cshift.data());        

        THRUST_CALL(thrust::inclusive_scan, dm, stream, offset_cshift.data(), offset_cshift.data() + num_positions, offset_cshift.data());

        cudaMemcpyAsync(&num_queries, offset_cshift.data() + num_positions - 1, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        query_target_indexes.resize(num_queries, stream);
        query_target_cell_shifts.resize(num_queries, stream);
        grid_size.x = (num_queries + block_size.x - 1) / block_size.x;
        get_cell_shifts_kernel<FloatType, IndexType><<<grid_size, block_size, 0, stream>>>(
            frac_radius, positions.data(), num_positions, offset_cshift.data(), 
            query_target_indexes.data(), query_target_cell_shifts.data(), num_queries);
    }

    // cuda_region_query_t implementations
    template <typename MortonCodeType, typename IndexType, typename FloatType>
    cuda_region_query_t<MortonCodeType, IndexType, FloatType>::cuda_region_query_t(
        const vector<MortonCodeType>& h_morton_codes, const IndexType num_bits_per_dim, 
        const vector<IndexType>& h_offsets, const vector<IndexType>& h_sorted_indexes, 
        cudaStream_t stream) 
        : offsets(h_offsets.size(), stream),
        sorted_indexes(h_sorted_indexes.size(), stream),
        brt(std::make_unique<cuda_binary_radix_tree_t<MortonCodeType, IndexType>>(h_morton_codes, num_bits_per_dim * 3)),
        num_bits_per_dim(num_bits_per_dim)
    {
        // copy data from host to device 
        cudaMemcpyAsync(offsets.data(), h_offsets.data(), h_offsets.size() * sizeof(IndexType), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(sorted_indexes.data(), h_sorted_indexes.data(), h_sorted_indexes.size() * sizeof(IndexType), cudaMemcpyHostToDevice, stream);
    }

    template class cuda_region_query_t<uint32_t, int32_t, float>;
    template class cuda_region_query_t<uint32_t, int32_t, double>;

    template <typename MortonCodeType, typename FloatType, typename IndexType>
    __global__
    void cuda_traverse_sphere_kernel(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, 
        const IndexType num_leaf_nodes, const cuda_check_sphere_t<MortonCodeType, FloatType, IndexType> check_method,
        const point3d_t<FloatType>* positions, 
        const IndexType* query_target_indexes, const array3d_t<IndexType>* query_target_cell_shifts,
        const IndexType num_queries, 
        IndexType* indexes, IndexType* num_indexes, const IndexType* indexes_offset = nullptr)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_queries) return;

        auto query_target_index = query_target_indexes[tid];
        auto position = positions[query_target_index];
        auto cell_shift = query_target_cell_shifts[tid];

        cuda_tree_traverse<cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>, MortonCodeType, FloatType, IndexType>(
            internal_nodes_tex, leaf_nodes_tex, num_leaf_nodes, check_method, 
            position, cell_shift, indexes, num_indexes[tid], 
            indexes_offset ? (tid > 0 ? indexes_offset[tid - 1] : 0) : 0);
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    __global__
    void filter_traverse_results_kernel(const IndexType* atom_offsets, const IndexType* atom_sorted_indexes, const point3d_t<FloatType>* positions, 
        const IndexType* query_target_indexes, const array3d_t<IndexType>* query_target_cell_shifts, const IndexType num_queries, 
        const atom_t<FloatType>* atoms, const sym_matrix3d_t<FloatType> metric, const matrix3d_t<FloatType> lattice_vectors, const FloatType radius2, 
        const IndexType* traverse_result_offset, const IndexType num_traverse_results, 
        const IndexType* indexes_per_traverse, 
        IndexType* query_offsets, cuda_query_result_t<FloatType>* query_results)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_traverse_results) return;

        // get query index and cell index for the traverse result
        auto query_index = gmp::util::binary_search_first_larger(traverse_result_offset, static_cast<IndexType>(0), num_queries - 1, static_cast<IndexType>(tid));
        auto cell_index = indexes_per_traverse[tid];

        // get the atoms in the cell
        auto start = atom_offsets[cell_index];
        auto end = atom_offsets[cell_index + 1];

        // get the query target
        auto query_target_index = query_target_indexes[query_index];
        auto position = positions[query_target_index];
        auto cell_shift = array3d_t<FloatType>{static_cast<FloatType>(query_target_cell_shifts[query_index][0]), 
                                               static_cast<FloatType>(query_target_cell_shifts[query_index][1]), 
                                               static_cast<FloatType>(query_target_cell_shifts[query_index][2])};

        auto shift = tid == 0 ? 0 : query_offsets[tid - 1];
        IndexType count = 0;
        for (auto idx = start; idx < end; idx++) {
            auto atom_index = atom_sorted_indexes[idx];
            auto atom_position = atoms[atom_index].pos;
            array3d_t<FloatType> difference;
            auto distance2 = gmp::util::cuda_calculate_distance_squared(metric, atom_position, position, cell_shift, difference);
            if (distance2 >= radius2) continue;
            if (query_results != nullptr) {
                array3d_t<FloatType> difference_cartesian = gmp::util::cuda_fractional_to_cartesian(lattice_vectors, difference);
                query_results[shift + count] = {difference_cartesian, distance2, atom_index};
            }
            count++;
        }
        
        if (query_results == nullptr) {
            query_offsets[tid] = count;
        }
    }

    template <typename IndexType> 
    __global__
    void compact_query_results_kernel(const size_t num_query_points, const IndexType* query_target_indexes, const IndexType num_queries, 
        const IndexType* query_offsets, const IndexType* traverse_result_offset, IndexType* compacted_query_offsets)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_query_points) return;

        auto query_point_index = gmp::util::binary_search_first_larger(
            query_target_indexes, static_cast<IndexType>(0), num_queries - 1, static_cast<IndexType>(tid)) - 1;
        
        compacted_query_offsets[tid] = query_offsets[traverse_result_offset[query_point_index] - 1];
    }

    template <typename FloatType, typename IndexType>
    struct compare_query_result_t {
        __host__ __device__
        bool operator()(const thrust::tuple<IndexType,cuda_query_result_t<FloatType>>& a, 
            const thrust::tuple<IndexType,cuda_query_result_t<FloatType>>& b) const 
        {
            IndexType sa = thrust::get<0>(a), sb = thrust::get<0>(b);
            cuda_query_result_t<FloatType> va = thrust::get<1>(a), vb = thrust::get<1>(b);
            return (sa < sb) || (sa == sb && (va.distance_squared < vb.distance_squared 
                || (va.distance_squared == vb.distance_squared && va.neighbor_index < vb.neighbor_index)));
        }
    };

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    void cuda_region_query(const vector_device<point3d_t<FloatType>>& positions, 
        const FloatType cutoff, const cuda_region_query_t<MortonCodeType, IndexType, FloatType>& region_query, 
        const lattice_t<FloatType>* lattice, const vector_device<atom_t<FloatType>>& atoms,
        vector_device<cuda_query_result_t<FloatType>>& query_results, vector_device<IndexType>& query_offsets,
        cudaStream_t stream)
    {
        auto dm = gmp::resources::gmp_resource::instance().get_device_memory_manager();
        dim3 block_size(256, 1, 1), grid_size(1, 1, 1);

        // get fractional radius
        auto frac_radius = cutoff / lattice->get_cell_lengths();

        // get query targets
        IndexType num_queries;
        vector_device<IndexType> query_target_indexes(1, stream);
        vector_device<array3d_t<IndexType>> query_target_cell_shifts(1, stream);
        get_query_targets(frac_radius, positions, num_queries, query_target_indexes, query_target_cell_shifts);

        // create check method 
        cuda_check_sphere_t<MortonCodeType, FloatType, IndexType> check_method;
        check_method.radius2 = cutoff * cutoff;
        check_method.num_bits_per_dim = region_query.num_bits_per_dim;
        check_method.metric = lattice->get_metric();

        // create traverse result
        cuda_traverse_result_t<IndexType> traverse_result(num_queries, stream);

        // launch kernel 
        grid_size.x = (num_queries + block_size.x - 1) / block_size.x;
        cuda_traverse_sphere_kernel<MortonCodeType, FloatType, IndexType><<<grid_size, block_size, 0, stream>>>(
            region_query.brt->internal_nodes_tex, region_query.brt->leaf_nodes_tex, region_query.brt->num_leaf_nodes, 
            check_method, positions.data(), query_target_indexes.data(), query_target_cell_shifts.data(), num_queries,
            nullptr, traverse_result.num_indexes.data()
        );

        // inclusive sum to get traverse result offset
        THRUST_CALL(thrust::inclusive_scan, dm, stream, 
            traverse_result.num_indexes.data(), traverse_result.num_indexes.data() + num_queries, traverse_result.num_indexes_offset.data());

        // get the total number of traverse results
        IndexType num_traverse_results;
        cudaMemcpyAsync(&num_traverse_results, traverse_result.num_indexes_offset.data() + num_queries - 1, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // zero out num indexes
        cudaMemsetAsync(traverse_result.num_indexes.data(), 0, num_queries * sizeof(IndexType), stream);

        // allocate memory for traverse result
        traverse_result.indexes.resize(num_traverse_results, stream);

        // second traverse to get traverse result
        cuda_traverse_sphere_kernel<MortonCodeType, FloatType, IndexType><<<grid_size, block_size, 0, stream>>>(
            region_query.brt->internal_nodes_tex, region_query.brt->leaf_nodes_tex, region_query.brt->num_leaf_nodes, 
            check_method, positions.data(), query_target_indexes.data(), query_target_cell_shifts.data(), num_queries,
            traverse_result.indexes.data(), traverse_result.num_indexes.data(), traverse_result.num_indexes_offset.data()
        );

        // filter over atoms per cell and count the number of atoms per traverse
        vector_device<IndexType> query_counts(num_traverse_results, stream);
        grid_size.x = (num_traverse_results + block_size.x - 1) / block_size.x;
        filter_traverse_results_kernel<MortonCodeType, IndexType, FloatType><<<grid_size, block_size, 0, stream>>>(
            region_query.offsets.data(), region_query.sorted_indexes.data(), 
            positions.data(), query_target_indexes.data(), query_target_cell_shifts.data(), num_queries, 
            atoms.data(), lattice->get_metric(), lattice->get_lattice_vectors(), check_method.radius2, 
            traverse_result.num_indexes_offset.data(), num_traverse_results, 
            traverse_result.indexes.data(),
            query_counts.data(), nullptr);

        // inclusive sum to get query offsets
        THRUST_CALL(thrust::inclusive_scan, dm, stream, 
            query_counts.data(), query_counts.data() + num_traverse_results, query_counts.data());

        // get the total number of query results
        IndexType num_query_results;
        cudaMemcpyAsync(&num_query_results, query_counts.data() + num_traverse_results - 1, sizeof(IndexType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // resize query results
        query_results.resize(num_query_results, stream);

        // get the atoms information 
        filter_traverse_results_kernel<MortonCodeType, IndexType, FloatType><<<grid_size, block_size, 0, stream>>>(
            region_query.offsets.data(), region_query.sorted_indexes.data(), 
            positions.data(), query_target_indexes.data(), query_target_cell_shifts.data(), num_queries, 
            atoms.data(), lattice->get_metric(), lattice->get_lattice_vectors(), check_method.radius2, 
            traverse_result.num_indexes_offset.data(), num_traverse_results, 
            traverse_result.indexes.data(),
            query_counts.data(), query_results.data());
        
        // compaction of query counts
        query_offsets.resize(positions.size(), stream);
        grid_size.x = (positions.size() + block_size.x - 1) / block_size.x;
        compact_query_results_kernel<<<grid_size, block_size, 0, stream>>>(
            positions.size(), query_target_indexes.data(), num_queries,
            query_counts.data(), traverse_result.num_indexes_offset.data(), query_offsets.data());
        
        // block sort the query results
        thrust_ops::segmented_sort_inplace(query_results, query_offsets, compare_query_result_t<FloatType, IndexType>(), dm, stream);

    }

    template 
    void cuda_region_query<uint32_t, int32_t, float>(
        const vector_device<point3d_t<float>>& positions,
        const float cutoff, const cuda_region_query_t<uint32_t, int32_t, float>& region_query, 
        const lattice_t<float>* lattice, const vector_device<atom_t<float>>& atoms,
        vector_device<cuda_query_result_t<float>>& query_results, 
        vector_device<int32_t>& query_offsets, cudaStream_t stream);
    
    template 
    void cuda_region_query<uint32_t, int32_t, double>(
        const vector_device<point3d_t<double>>& positions,
        const double cutoff, const cuda_region_query_t<uint32_t, int32_t, double>& region_query, 
        const lattice_t<double>* lattice, const vector_device<atom_t<double>>& atoms,
        vector_device<cuda_query_result_t<double>>& query_results, 
        vector_device<int32_t>& query_offsets, cudaStream_t stream);

}} // namespace gmp::region_query