#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include "morton_codes.hpp"
#include "containers.hpp"
#include "math.hpp"
#include "common_types.hpp"
#include "resources.hpp"
#include "geometry.hpp"

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
        IndexType max_num_mc; // maximum number of mc to be stored
        IndexType num_queries; // number of queries

        cuda_traverse_result_t(IndexType max_num_mc, IndexType num_queries, cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream()) : 
            indexes(max_num_mc * num_queries, stream),
            num_indexes(num_queries, stream),
            max_num_mc(max_num_mc),
            num_queries(num_queries)
        {
            cudaMemsetAsync(indexes.data(), 0, max_num_mc * num_queries * sizeof(IndexType), stream);
            cudaMemsetAsync(num_indexes.data(), 0, num_queries * sizeof(IndexType), stream);
        }
        ~cuda_traverse_result_t() = default;
    };
    
    // check intersect box class for test purpose
    template <typename MortonCodeType, typename FloatType, typename IndexType>
    struct cuda_check_intersect_box_t 
    {
        MortonCodeType query_lower_bound, query_upper_bound;
        MortonCodeType x_mask, y_mask, z_mask;

        __device__
        bool operator()(const MortonCodeType lower_bound, const MortonCodeType upper_bound, 
            const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift) const
        {
            return mc_is_less_than_or_equal(query_lower_bound, upper_bound, x_mask, y_mask, z_mask) && 
                    mc_is_less_than_or_equal(lower_bound, query_upper_bound, x_mask, y_mask, z_mask);
        }

        __device__
        void operator() (const MortonCodeType morton_code, const IndexType idx, 
            const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift, 
            IndexType* indexes, IndexType* num_indexes) const
        {
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
        IndexType num_bits_per_dim;
        sym_matrix3d_t<FloatType> metric;
        // const array3d_bool periodicity;

        __device__
        bool operator()(const MortonCodeType lower_bound, const MortonCodeType upper_bound, 
            const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift) const;

        __device__
        void operator()(const MortonCodeType morton_code, const IndexType idx, 
            const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift, 
            IndexType* indexes, IndexType* num_indexes) const;
    };

    template <
        typename MortonCodeType = std::uint32_t, 
        typename IndexType = std::int32_t
    >
    class cuda_binary_radix_tree_t {
        using inode_t = internal_node_t<MortonCodeType, IndexType>;

    public:
        cuda_binary_radix_tree_t(const vector<MortonCodeType>& morton_codes, const IndexType num_bits, cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream());
        ~cuda_binary_radix_tree_t();
        
        void get_internal_nodes(vector<inode_t>& h_internal_nodes) const;
        void get_leaf_nodes(vector<MortonCodeType>& h_leaf_nodes) const;

        IndexType num_leaf_nodes;
        vector_device<inode_t> internal_nodes;
        vector_device<MortonCodeType> leaf_nodes;
        
        // Texture memory for tree data
        cudaTextureObject_t internal_nodes_tex;
        cudaTextureObject_t leaf_nodes_tex;
    };

    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType>
    __device__
    void cuda_tree_traverse(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, const IndexType num_leaf_nodes, 
        const Checker check_method, const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift, IndexType* indexes, IndexType* num_indexes);

    // Texture memory setup and teardown
    void bind_texture_memory(void* data_ptr, uint32_t size, int bits_per_channel, cudaTextureObject_t& tex);
    void unbind_texture_memory(cudaTextureObject_t tex);

}} // namespace gmp::tree
