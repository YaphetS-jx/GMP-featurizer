#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include "morton_codes.hpp"
#include "containers.hpp"
#include "common_types.hpp"

namespace gmp { namespace tree {
    
    using gmp::containers::vector;
    using gmp::math::array3d_int32;
    using namespace morton_codes;
    
    template <
        typename MortonCodeType = std::uint32_t, 
        typename IndexType = std::int32_t,
        template<typename, typename...> class Container = vector, 
        typename map_t = std::unordered_map<IndexType, vector<array3d_int32>>
    >
    class binary_radix_tree_t {
        using inode_t = internal_node_t<MortonCodeType, IndexType>;
        using morton_container_t = Container<MortonCodeType>;
        using node_container_t = Container<inode_t>;
    
    public: 
        binary_radix_tree_t();
        binary_radix_tree_t(const morton_container_t& morton_codes, const IndexType num_bits = 30);
        const node_container_t& get_internal_nodes() const;
        const morton_container_t& get_leaf_nodes() const;        
        void build_tree(const morton_container_t& morton_codes, const IndexType num_bits = 30);
        map_t traverse(const compare_op_t<MortonCodeType> &check_method) const;
    private:
        node_container_t internal_nodes;
        morton_container_t leaf_nodes;
    };

}} // namespace gmp::tree
