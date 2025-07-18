#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include "morton_codes.hpp"
#include "containers.hpp"
#include "math.hpp"

namespace gmp { namespace tree {
    
    using namespace morton_codes;
    // using namespace gmp::containers;
    // using namespace gmp::math;
    using gmp::containers::vector;
    using gmp::containers::stack;
    using gmp::math::array3d_int32;

    template <typename MortonCodeType = std::uint32_t, typename IndexType = std::int32_t>
    struct internal_node_t {
        IndexType left, right;
        MortonCodeType lower_bound, upper_bound;
        internal_node_t(IndexType left, IndexType right, MortonCodeType lower_bound, MortonCodeType upper_bound)
            : left(left), right(right), lower_bound(lower_bound), upper_bound(upper_bound) {}
    };

    template <typename MortonCodeType, typename VecType = vector<array3d_int32>>
    class compare_op_t {
    public: 
        virtual bool operator()(MortonCodeType query_lower_bound, MortonCodeType query_upper_bound) const = 0;
        virtual VecType operator()(MortonCodeType morton_code) const = 0;
    };

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
        IndexType count_leading_zeros(MortonCodeType x, IndexType num_bits) const;
        IndexType delta(const morton_container_t& morton_codes, IndexType i, IndexType j, IndexType num_bits = 30) const;
        void determine_range(const morton_container_t& morton_codes, IndexType i, IndexType& first, IndexType& last, IndexType num_bits = 30) const;
        IndexType find_split(const morton_container_t& morton_codes, IndexType delta_node, IndexType first, IndexType last, IndexType num_bits) const;
        void find_lower_upper_bounds(MortonCodeType prefix, IndexType num_bits_prefix, MortonCodeType& lower_bound, MortonCodeType& upper_bound, IndexType num_bits = 30) const;
        void build_tree(const morton_container_t& morton_codes, const IndexType num_bits = 30);
        map_t traverse(const compare_op_t<MortonCodeType> &check_method) const;
    private:
        node_container_t internal_nodes;
        morton_container_t leaf_nodes;
    };

}} // namespace gmp::tree
