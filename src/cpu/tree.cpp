#include "tree.hpp"

namespace gmp { namespace tree {

    using gmp::containers::stack;

    // binary_radix_tree_t implementation
    template <typename MortonCodeType, typename IndexType>
    binary_radix_tree_t<MortonCodeType, IndexType>::binary_radix_tree_t() {}

    template <typename MortonCodeType, typename IndexType>
    binary_radix_tree_t<MortonCodeType, IndexType>::binary_radix_tree_t(const morton_container_t& morton_codes, const IndexType num_bits)
    {
        build_tree(morton_codes, num_bits);
    }

    template <typename MortonCodeType, typename IndexType>
    const typename binary_radix_tree_t<MortonCodeType, IndexType>::node_container_t&
    binary_radix_tree_t<MortonCodeType, IndexType>::get_internal_nodes() const
    {
        return internal_nodes;
    }

    template <typename MortonCodeType, typename IndexType>
    const typename binary_radix_tree_t<MortonCodeType, IndexType>::morton_container_t&
    binary_radix_tree_t<MortonCodeType, IndexType>::get_leaf_nodes() const
    {
        return leaf_nodes;
    }

    template <typename MortonCodeType, typename IndexType>
    void binary_radix_tree_t<MortonCodeType, IndexType>::build_tree(const morton_container_t& morton_codes, const IndexType num_bits)
    {
        assert(num_bits % 3 == 0);
        auto n = static_cast<IndexType>(morton_codes.size());
        internal_nodes.reserve(n - 1);

        for (IndexType i = 0; i < n - 1; ++i) {
            IndexType first, last;
            morton_codes::determine_range(morton_codes.data(), n, i, first, last, num_bits);
            IndexType delta_node = morton_codes::delta(morton_codes.data(), n, first, last, num_bits);
            IndexType split = morton_codes::find_split(morton_codes.data(), n, delta_node, first, last, num_bits);
            MortonCodeType lower_bound, upper_bound;
            morton_codes::find_lower_upper_bounds(morton_codes[split], delta_node, lower_bound, upper_bound, num_bits);

            // Determine left and right children
            IndexType left = (split == first) ? split : split + n;
            IndexType right = (split + 1 == last) ? split + 1 : split + 1 + n;

            internal_nodes.push_back({left, right, lower_bound, upper_bound});
        }
        // save leaf nodes
        leaf_nodes = std::move(morton_codes);
    }

    template <typename MortonCodeType, typename IndexType>
    typename binary_radix_tree_t<MortonCodeType, IndexType>::map_t binary_radix_tree_t<MortonCodeType, IndexType>::traverse(const compare_op_t<MortonCodeType> &check_method) const
    {
        map_t result;
        stack<IndexType> s;
        auto n = leaf_nodes.size();
        s.push(n);
        while (!s.empty()) {
            IndexType node_index = s.top();
            s.pop();
            
            if (node_index < n) {
                auto temp = check_method(leaf_nodes[node_index]);
                if (!temp.empty()) {
                    result[node_index] = std::move(temp);
                }
            } else {
                const auto& node = internal_nodes[node_index - n];
                if (check_method(node.lower_bound, node.upper_bound)) {
                    s.push(node.left);
                    s.push(node.right);
                }
            }
        }
        return result;
    }

    // Explicit instantiations for binary_radix_tree_t (used in tests)
    template class binary_radix_tree_t<uint32_t, int32_t>;

}} // namespace gmp::tree 