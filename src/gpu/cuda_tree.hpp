#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include "morton_codes.hpp"
#include "containers.hpp"
#include "math.hpp"
#include "common_types.hpp"
#include "resources.hpp"

namespace gmp { namespace tree {
    
    using namespace morton_codes;
    using gmp::containers::vector_device;
    using gmp::containers::vector_host;
    using gmp::math::array3d_t;

    template <typename IndexType>
    class traverse_result_t {
    public:
        vector_device<IndexType> indexes; // preallocated by overestimation 
        vector_device<size_t> num_indexes; // number of indexes saved
        size_t max_num_mc; // maximum number of mc to be stored
        size_t num_queries; // number of queries

        traverse_result_t(size_t max_num_mc, size_t num_queries, cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream()) : 
            indexes(max_num_mc * num_queries, stream),
            num_indexes(num_queries, stream),
            max_num_mc(max_num_mc),
            num_queries(num_queries)
        {}
        ~traverse_result_t() = default;
    };
    
    template <typename MortonCodeType, typename IndexType>
    class cuda_compare_op_t {
    public: 
        __device__
        virtual bool operator()(MortonCodeType query_lower_bound, MortonCodeType query_upper_bound, 
            const array3d_t<IndexType>& cell_shifts) const = 0;
        __device__
        virtual void operator()(MortonCodeType morton_code, const array3d_t<IndexType>& cell_shifts, IndexType idx, 
            IndexType* indexes, size_t* num_indexes) const = 0;
    };

    template <
        typename MortonCodeType = std::uint32_t, 
        typename IndexType = std::int32_t
    >
    class cuda_binary_radix_tree_t {
        using inode_t = internal_node_t<MortonCodeType, IndexType>;

    public:
        cuda_binary_radix_tree_t(const vector_device<MortonCodeType>& morton_codes, const IndexType num_bits, cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream());
        ~cuda_binary_radix_tree_t();
        
        void get_internal_nodes(vector_host<inode_t>& h_internal_nodes) const;
        void get_leaf_nodes(vector_host<MortonCodeType>& h_leaf_nodes) const;

        IndexType num_leaf_nodes;
        vector_device<inode_t> internal_nodes;
        vector_device<MortonCodeType> leaf_nodes;
        
        // Texture memory for tree data
        cudaTextureObject_t internal_nodes_tex;
        cudaTextureObject_t leaf_nodes_tex;
    };

    template <typename MortonCodeType, typename IndexType>
    __device__
    void tree_traverse_kernel(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, 
        const IndexType num_leaf_nodes, const cuda_compare_op_t<MortonCodeType, IndexType>& check_method,
        const array3d_t<IndexType>& cell_shifts, IndexType* indexes, size_t* num_indexes);

    // Texture memory setup and teardown
    void bind_texture_memory(void* data_ptr, size_t size, int bits_per_channel, cudaTextureObject_t& tex);
    void unbind_texture_memory(cudaTextureObject_t tex);

}} // namespace gmp::tree
