#include "tree.hpp"

namespace gmp { namespace tree {

    // binary_radix_tree_t implementation
    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    IndexType
    binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::count_leading_zeros(MortonCodeType x, IndexType num_bits) const
    {
        return num_bits - (sizeof(MortonCodeType) * 8 - __builtin_clz(x));
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::binary_radix_tree_t() {}

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::binary_radix_tree_t(const morton_container_t& morton_codes, const IndexType num_bits)
    {
        build_tree(morton_codes, num_bits);
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    const typename binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::node_container_t&
    binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::get_internal_nodes() const
    {
        return internal_nodes;
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    const typename binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::morton_container_t&
    binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::get_leaf_nodes() const
    {
        return leaf_nodes;
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    IndexType
    binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::delta(const morton_container_t& morton_codes, IndexType i, IndexType j, IndexType num_bits) const
    {
        assert(i >= 0 && i < morton_codes.size());
        if (j < 0 || j >= morton_codes.size()) {
            return static_cast<IndexType>(-1);
        }
        return static_cast<IndexType>(count_leading_zeros(morton_codes[i] ^ morton_codes[j], num_bits));
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    void binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::determine_range(const morton_container_t& morton_codes, IndexType i, IndexType& first, IndexType& last, IndexType num_bits) const
    {
        assert(i >= 0 && i < morton_codes.size());
        // Calculate direction based on delta differences         
        IndexType delta_prev = delta(morton_codes, i, i - 1, num_bits);
        IndexType delta_next = delta(morton_codes, i, i + 1, num_bits);
        IndexType d = (delta_next > delta_prev) ? 1 : -1;
        IndexType delta_min = delta(morton_codes, i, i - d, num_bits);            

        // Exponential search to find other end
        IndexType lmax = 2;
        while (delta(morton_codes, i, i + lmax * d, num_bits) > delta_min) {
            lmax *= 2;
        }

        // Binary search to refine
        IndexType l = 0;
        IndexType t = lmax / 2;
        while (t >= 1) {
            IndexType next = i + (l + t) * d;
            if (delta(morton_codes, i, next, num_bits) > delta_min) {
                l += t;
            }
            t /= 2;
        }

        IndexType j = i + l * d;
        first = std::min(i, j);
        last = std::max(i, j);
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    IndexType
    binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::find_split(const morton_container_t& morton_codes, IndexType delta_node, IndexType first, IndexType last, IndexType num_bits) const
    {
        assert(first >= 0 && first < morton_codes.size());
        assert(last >= 0 && last < morton_codes.size());                         
        
        // Binary search for the split point
        IndexType split = first;
        IndexType stride = last - first;
        
        while (stride > 1) {
            stride = (stride + 1) / 2;
            IndexType mid = split + stride;
            
            if (mid < last && delta(morton_codes, first, mid, num_bits) > delta_node) {
                split = mid;
            }
        }
        return split;
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    void binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::find_lower_upper_bounds(MortonCodeType prefix, IndexType num_bits_prefix, MortonCodeType& lower_bound, MortonCodeType& upper_bound, IndexType num_bits) const
    {
        MortonCodeType mask1 = ((1 << num_bits_prefix) - 1) << (num_bits - num_bits_prefix);
        MortonCodeType mask2 = mask1 ^ ((1 << num_bits) - 1);

        lower_bound = prefix & mask1;
        upper_bound = prefix | mask2;
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    void binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::build_tree(const morton_container_t& morton_codes, const IndexType num_bits)
    {
        assert(num_bits % 3 == 0);
        auto n = static_cast<IndexType>(morton_codes.size());
        internal_nodes.reserve(n - 1);

        for (IndexType i = 0; i < n - 1; ++i) {
            IndexType first, last;
            determine_range(morton_codes, i, first, last, num_bits);
            IndexType delta_node = delta(morton_codes, first, last, num_bits);
            IndexType split = find_split(morton_codes, delta_node, first, last, num_bits);
            MortonCodeType lower_bound, upper_bound;
            find_lower_upper_bounds(morton_codes[split], delta_node, lower_bound, upper_bound, num_bits);

            // Determine left and right children
            IndexType left = (split == first) ? split : split + n;
            IndexType right = (split + 1 == last) ? split + 1 : split + 1 + n;

            internal_nodes.emplace_back(left, right, lower_bound, upper_bound);
        }
        // save leaf nodes
        leaf_nodes = std::move(morton_codes);
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container, typename map_t>
    map_t binary_radix_tree_t<MortonCodeType, IndexType, Container, map_t>::traverse(const compare_op_t<MortonCodeType> &check_method) const
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