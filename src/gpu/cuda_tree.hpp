#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include "morton_codes.hpp"
#include "containers.hpp"
#include "math.hpp"
#include "resources.hpp"
#include "geometry.hpp"
#include "tree.hpp"

namespace gmp { namespace tree {

    using namespace morton_codes;
    using gmp::containers::vector_device;
    using gmp::containers::vector;
    using gmp::geometry::point3d_t;
    using gmp::math::array3d_t;
    using gmp::math::sym_matrix3d_t;
    
    template <typename IndexType>
    class cuda_traverse_result_t {
    public:
        vector_device<IndexType> indexes; // preallocated by overestimation 
        vector_device<IndexType> num_indexes; // number of indexes saved
        vector_device<IndexType> num_indexes_offset; // offset of num_indexes
        IndexType num_queries; // number of queries

        cuda_traverse_result_t(IndexType num_queries, cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream()) : 
            indexes(1, stream),
            num_indexes(num_queries, stream),
            num_indexes_offset(num_queries, stream),
            num_queries(num_queries)
        {
            cudaMemsetAsync(num_indexes.data(), 0, num_queries * sizeof(IndexType), stream);
        }
        ~cuda_traverse_result_t() = default;
    };
    
    // check intersect box class for test purpose
    template <typename MortonCodeType, typename FloatType, typename IndexType>
    struct cuda_check_intersect_box_t 
    {
        IndexType num_bits_per_dim;
        MortonCodeType query_lower_bound, query_upper_bound;
        MortonCodeType x_mask, y_mask, z_mask;

        __device__
        bool operator()(const array3d_t<FloatType>& lower_coords, const array3d_t<FloatType>& upper_coords, 
            const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift) const
        {
            return true;
        }

        __device__
        void operator() (const array3d_t<FloatType> leaf_coords, const IndexType idx, 
            const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift, 
            IndexType* indexes, IndexType* num_indexes, const IndexType indexes_offset = 0) const /* indexes_offset is deprecated for box intersection check */
        {
            if (indexes == nullptr) return;
            // Reconstruct morton code from coordinates to check bounds
            uint32_t mc_x = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(leaf_coords[0], num_bits_per_dim);
            uint32_t mc_y = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(leaf_coords[1], num_bits_per_dim);
            uint32_t mc_z = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(leaf_coords[2], num_bits_per_dim);
            uint32_t morton_code = interleave_bits(mc_x, mc_y, mc_z, num_bits_per_dim);
        
            if (mc_is_less_than_or_equal(query_lower_bound, morton_code, x_mask, y_mask, z_mask) && 
                mc_is_less_than_or_equal(morton_code, query_upper_bound, x_mask, y_mask, z_mask)) 
            {
                indexes[*num_indexes] = idx;
                (*num_indexes)++;
            }
        }
    };

    // check sphere class
    template <typename MortonCodeType, typename FloatType, typename IndexType>
    struct cuda_check_sphere_t {
        FloatType radius2;
        FloatType size_per_dim;
        IndexType num_bits_per_dim;
        sym_matrix3d_t<FloatType> metric;
        // const array3d_bool periodicity;

        __device__
        bool operator()(const array3d_t<FloatType>& lower_coords, const array3d_t<FloatType>& upper_coords, 
            const point3d_t<FloatType>& position, const array3d_t<IndexType>& cell_shift) const;

        __device__
        void operator()(const array3d_t<FloatType> leaf_coords, const IndexType idx, 
            const point3d_t<FloatType>& position, const array3d_t<IndexType>& cell_shift, 
            IndexType* indexes, IndexType* num_indexes, const IndexType indexes_offset) const;
    };

    extern __constant__ cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t> check_sphere_constant;

    template <
        typename IndexType = std::int32_t,
        typename FloatType = gmp::gmp_float
    >
    class cuda_binary_radix_tree_t {
        using inode_t = internal_node_t<IndexType, FloatType>;

    public:
        template <typename MortonCodeType = std::uint32_t>
        cuda_binary_radix_tree_t(const vector<MortonCodeType>& morton_codes, const IndexType num_bits, cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream());
        
        void get_internal_nodes(vector<inode_t>& h_internal_nodes) const;
        void get_leaf_nodes(vector<array3d_t<FloatType>>& h_leaf_nodes) const;

        IndexType num_leaf_nodes;
        vector_device<inode_t> internal_nodes;
        vector_device<array3d_t<FloatType>> leaf_nodes;
    };

    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType>
    __device__
    void cuda_tree_traverse(const internal_node_t<IndexType, FloatType>* internal_nodes, const array3d_t<FloatType>* leaf_nodes, const IndexType num_leaf_nodes, 
        const Checker check_method, const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift, IndexType* indexes, IndexType* num_indexes, const IndexType indexes_offset = 0);

    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType, int MAX_STACK=64>
    __global__
    void cuda_tree_traverse_warp(const internal_node_t<IndexType, FloatType>* internal_nodes, const array3d_t<FloatType>* leaf_nodes, const IndexType num_leaf_nodes, 
        const Checker& check_method, const point3d_t<FloatType>* positions, const IndexType* query_target_indexes,
        const array3d_t<IndexType>* cell_shifts, const IndexType num_queries,
        IndexType* indexes, IndexType* num_indexes, const IndexType* num_indexes_offset);

    template <typename MortonCodeType, typename FloatType, typename IndexType, int MAX_STACK>
    __global__
    void cuda_tree_traverse_warp2(const internal_node_t<IndexType, FloatType>* internal_nodes, const array3d_t<FloatType>* leaf_nodes, const IndexType num_leaf_nodes, 
        const point3d_t<FloatType>* positions, const IndexType* query_target_indexes,
        const array3d_t<IndexType>* cell_shifts, const IndexType num_queries,
        IndexType* indexes, IndexType* num_indexes, const IndexType* num_indexes_offset);
}} // namespace gmp::tree
