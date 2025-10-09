#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include "morton_codes.hpp"
#include "containers.hpp"
#include "math.hpp"
#include "gmp_float.hpp"

namespace gmp { namespace tree {
    
    using gmp::containers::vector;
    using gmp::math::array3d_int32;
    using gmp::math::array3d_t;
    using namespace morton_codes;

    template <typename IndexType = std::int32_t, typename FloatType = gmp::gmp_float>
    struct internal_node_t {
        // Precomputed float coordinates for optimization
        // left and right indices are stored in the padding_ field of lower_bound_coords
        array3d_t<FloatType> lower_bound_coords;
        array3d_t<FloatType> upper_bound_coords;
        
        // Accessor methods for left and right indices
        GPU_HOST_DEVICE void set_indices(IndexType left, IndexType right) {
            lower_bound_coords.template set_extra<IndexType>(left);
            upper_bound_coords.template set_extra<IndexType>(right);
        }
        
        GPU_HOST_DEVICE IndexType get_left() const {
            return lower_bound_coords.template get_extra<IndexType>();
        }
        
        GPU_HOST_DEVICE IndexType get_right() const {
            return upper_bound_coords.template get_extra<IndexType>();
        }
    };

    template <typename FloatType = gmp::gmp_float>
    class compare_op_t {
    public: 
        virtual bool operator()(const array3d_t<FloatType>& lower_coords, const array3d_t<FloatType>& upper_coords) const = 0;
        virtual std::vector<array3d_int32> operator()(const array3d_t<FloatType>& lower_coords, FloatType size_per_dim) const = 0;
    };

    template <
        typename IndexType = std::int32_t,
        typename FloatType = gmp::gmp_float
    >
    class binary_radix_tree_t {
        using inode_t = internal_node_t<IndexType, FloatType>;
        using node_container_t = std::vector<inode_t>;
        using leaf_container_t = std::vector<array3d_t<FloatType>>;
        using map_t = std::unordered_map<IndexType, std::vector<array3d_int32>>;
    
    public: 
        binary_radix_tree_t();
        template <typename MortonCodeType = std::uint32_t>
        binary_radix_tree_t(const vector<MortonCodeType>& morton_codes, const IndexType num_bits = 30);
        const node_container_t& get_internal_nodes() const;
        const leaf_container_t& get_leaf_nodes() const;        
        template <typename MortonCodeType = std::uint32_t>
        void build_tree(const vector<MortonCodeType>& morton_codes, const IndexType num_bits = 30);
        map_t traverse(const compare_op_t<FloatType> &check_method) const;
    private:
        node_container_t internal_nodes;
        leaf_container_t leaf_nodes;
        IndexType num_bits_per_dim;
    };

}} // namespace gmp::tree
