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

    template <typename IndexType>
    struct traverse_result_t {
        IndexType* indexes; // preallocated by overestimation 
        uint32_t* shift;
        size_t num_indexes; // number of indexes saved
        array3d_int32* cell_shifts;
        uint32_t num_results; // number of results found        
        size_t max_num_mc; // maximum number of mc to be stored

        traverse_result_t(size_t max_num_mc_);
        ~traverse_result_t();
    };

    template <typename MortonCodeType, typename IndexType>
    class cuda_compare_op_t {
    public: 
        __device__
        virtual bool operator()(MortonCodeType query_lower_bound, MortonCodeType query_upper_bound) const = 0;
        __device__
        virtual void operator()(MortonCodeType morton_code, IndexType idx, 
            IndexType* indexes, uint32_t* count, size_t& num_indexes, array3d_int32* cell_shifts, uint32_t& num_results) const = 0;
    };

    template <
        typename MortonCodeType = std::uint32_t, 
        typename IndexType = std::int32_t
    >
    class cuda_binary_radix_tree_t {
        using inode_t = internal_node_t<MortonCodeType, IndexType>;

    public:
        cuda_binary_radix_tree_t() = default;
        ~cuda_binary_radix_tree_t() = default;
        
        void get_internal_nodes(inode_t* internal_nodes) const;
        void get_leaf_nodes(MortonCodeType* leaf_nodes) const;

    // private:
        inode_t* internal_nodes;
        MortonCodeType* leaf_nodes;
        size_t num_leaf_nodes;
        
        // Texture memory for tree data
        cudaTextureObject_t internal_nodes_tex;
        cudaTextureObject_t leaf_nodes_tex;
    };

    template <typename MortonCodeType, typename IndexType>
    __device__
    void tree_traverse_kernel(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, 
        const IndexType num_leaf_nodes, const cuda_compare_op_t<MortonCodeType, IndexType>& check_method, 
        IndexType* indexes, uint32_t* count, size_t& num_indexes, array3d_int32* cell_shifts, uint32_t& num_results);

    // function to build the tree
    template <typename MortonCodeType, typename IndexType>
    cuda_binary_radix_tree_t<MortonCodeType, IndexType> cuda_build_tree(const vector_device<MortonCodeType>& morton_codes, const IndexType num_bits);

    // Texture memory setup and teardown
    void bind_texture_memory(void* data_ptr, size_t size, int bits_per_channel, cudaTextureObject_t& tex);
    void unbind_texture_memory(cudaTextureObject_t tex);

}} // namespace gmp::tree
