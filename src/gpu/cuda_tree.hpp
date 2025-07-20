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
    
    using gmp::containers::vector_host;
    using gmp::containers::vector_device;
    using gmp::math::array3d_int32;
    using namespace morton_codes;

    template <
        typename MortonCodeType = std::uint32_t, 
        typename IndexType = std::int32_t,
        template<typename, typename...> class Container_host = vector_host,
        template<typename, typename...> class Container_device = vector_device
    >
    class cuda_binary_radix_tree_t {
        using inode_t = internal_node_t<MortonCodeType, IndexType>;
        using morton_container_host_t = Container_host<MortonCodeType>;
        using node_container_host_t = Container_host<inode_t>;
        using morton_container_device_t = Container_device<MortonCodeType>;
        using node_container_device_t = Container_device<inode_t>;
    
    public:
        cuda_binary_radix_tree_t(const morton_container_device_t& morton_codes, const IndexType num_bits, cudaStream_t stream);
        ~cuda_binary_radix_tree_t();
        node_container_host_t get_internal_nodes() const;
        morton_container_host_t get_leaf_nodes() const;        
        void build_tree(const morton_container_device_t& morton_codes, const IndexType num_bits, cudaStream_t stream);
        
        // New traverse method with two-pass approach
        struct traverse_result_t {
            vector_device<IndexType> indices;
            vector_device<int32_t> shifts; // Simple array of 3 int32s per result
            
            traverse_result_t() = default;
            traverse_result_t(vector_device<IndexType>&& idx, vector_device<int32_t>&& sh) 
                : indices(std::move(idx)), shifts(std::move(sh)) {}
        };
        
        traverse_result_t traverse(const MortonCodeType query_lower_bound, 
                                  const MortonCodeType query_upper_bound,
                                  cudaStream_t stream) const;
        
        // Texture memory setup
        void bind_texture_memory(cudaStream_t stream);
        void unbind_texture_memory();
        
    private:
        node_container_device_t internal_nodes;
        morton_container_device_t leaf_nodes;
        
        // Texture memory for tree data
        cudaTextureObject_t internal_nodes_tex;
        cudaTextureObject_t leaf_nodes_tex;
        bool textures_bound;
    };

    
    
}} // namespace gmp::tree
