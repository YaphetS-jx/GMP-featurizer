#include "tree.hpp"
#include "morton_codes.hpp"
#include "gmp_float.hpp"
#include <cassert>

namespace gmp { namespace tree {

    using gmp::containers::stack;
    using gmp::math::array3d_t;

    // binary_radix_tree_t implementation
    template <typename IndexType, typename FloatType>
    binary_radix_tree_t<IndexType, FloatType>::binary_radix_tree_t() {}

    template <typename IndexType, typename FloatType>
    template <typename MortonCodeType>
    binary_radix_tree_t<IndexType, FloatType>::binary_radix_tree_t(const vector<MortonCodeType>& morton_codes, const IndexType num_bits)
    {
        build_tree(morton_codes, num_bits);
    }

    template <typename IndexType, typename FloatType>
    const typename binary_radix_tree_t<IndexType, FloatType>::node_container_t&
    binary_radix_tree_t<IndexType, FloatType>::get_internal_nodes() const
    {
        return internal_nodes;
    }

    template <typename IndexType, typename FloatType>
    const typename binary_radix_tree_t<IndexType, FloatType>::leaf_container_t&
    binary_radix_tree_t<IndexType, FloatType>::get_leaf_nodes() const
    {
        return leaf_nodes;
    }

    template <typename IndexType, typename FloatType>
    template <typename MortonCodeType>
    void binary_radix_tree_t<IndexType, FloatType>::build_tree(const vector<MortonCodeType>& morton_codes, const IndexType num_bits)
    {
        assert(num_bits % 3 == 0);
        auto n = static_cast<IndexType>(morton_codes.size());
        internal_nodes.reserve(n - 1);

        // Calculate num_bits_per_dim for coordinate conversion
        this->num_bits_per_dim = num_bits / 3;
        FloatType size_per_dim = 1.0 / (1 << (num_bits_per_dim - 1));

        for (IndexType i = 0; i < n - 1; ++i) {
            IndexType first, last;
            morton_codes::determine_range(morton_codes.data(), n, i, first, last, num_bits);
            IndexType delta_node = morton_codes::delta(morton_codes.data(), n, first, last, num_bits);
            IndexType split = morton_codes::find_split(morton_codes.data(), n, delta_node, first, last, num_bits);
            MortonCodeType lower_bound, upper_bound;
            morton_codes::find_lower_upper_bounds(morton_codes[split], delta_node, lower_bound, upper_bound, num_bits);

            // Precompute float coordinates directly from morton codes
            MortonCodeType x_min, y_min, z_min;
            morton_codes::deinterleave_bits(lower_bound, num_bits_per_dim, x_min, y_min, z_min);
            MortonCodeType x_max, y_max, z_max;
            morton_codes::deinterleave_bits(upper_bound, num_bits_per_dim, x_max, y_max, z_max);

            array3d_t<FloatType> lower_bound_coords = {
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim),
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim),
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim)
            };

            array3d_t<FloatType> upper_bound_coords = {
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_max, num_bits_per_dim) + size_per_dim,
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_max, num_bits_per_dim) + size_per_dim,
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_max, num_bits_per_dim) + size_per_dim
            };

            // Determine left and right children
            IndexType left = (split == first) ? split : split + n;
            IndexType right = (split + 1 == last) ? split + 1 : split + 1 + n;

            internal_node_t<IndexType, FloatType> node;
            node.lower_bound_coords = lower_bound_coords;
            node.upper_bound_coords = upper_bound_coords;
            node.set_indices(left, right);
            internal_nodes.push_back(node);
        }
        
        // Create leaf nodes with precomputed coordinates
        leaf_nodes.clear();
        leaf_nodes.reserve(n);
        for (IndexType i = 0; i < n; ++i) {
            MortonCodeType morton_code = morton_codes[i];
            
            // Precompute float coordinates for leaf node
            MortonCodeType x_min, y_min, z_min;
            morton_codes::deinterleave_bits(morton_code, num_bits_per_dim, x_min, y_min, z_min);
            
            array3d_t<FloatType> lower_bound_coords = {
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim),
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim),
                morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim)
            };
            
            leaf_nodes.push_back(lower_bound_coords);
        }
    }

    template <typename IndexType, typename FloatType>
    typename binary_radix_tree_t<IndexType, FloatType>::map_t binary_radix_tree_t<IndexType, FloatType>::traverse(const compare_op_t<FloatType> &check_method) const
    {
        map_t result;
        stack<IndexType> s;
        auto n = leaf_nodes.size();
        s.push(n);
        while (!s.empty()) {
            IndexType node_index = s.top();
            s.pop();
            
            if (node_index < n) {
                const auto& leaf_coords = leaf_nodes[node_index];
                // Calculate size_per_dim on-the-fly
                FloatType size_per_dim = 1.0 / (1 << (this->num_bits_per_dim - 1));
                // Use the optimized operator with precomputed coordinates
                auto temp = check_method(leaf_coords, size_per_dim);
                if (!temp.empty()) {
                    result[node_index] = std::move(temp);
                }
            } else {
                const auto& node = internal_nodes[node_index - n];
                if (check_method(node.lower_bound_coords, node.upper_bound_coords)) {
                    s.push(node.get_left());
                    s.push(node.get_right());
                }
            }
        }
        return result;
    }

    // Explicit instantiations for binary_radix_tree_t (used in tests)
    template class binary_radix_tree_t<int32_t, gmp::gmp_float>;
    
    // Explicit instantiations for member function templates
    template binary_radix_tree_t<int32_t, gmp::gmp_float>::binary_radix_tree_t(const vector<uint32_t>&, int32_t);
    template void binary_radix_tree_t<int32_t, gmp::gmp_float>::build_tree(const vector<uint32_t>&, int32_t);
}} // namespace gmp::tree 