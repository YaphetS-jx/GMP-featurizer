#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include <bit>
#include "morton_codes.hpp"
#include "types.hpp"

namespace gmp { namespace tree {
    
    using namespace morton_codes;
    using namespace gmp::containers;

    template <typename MortonCodeType = std::uint32_t, typename IndexType = std::int32_t>
    struct internal_node_t {
        IndexType left, right;
        MortonCodeType lower_bound, upper_bound;
        internal_node_t(IndexType left, IndexType right, MortonCodeType lower_bound, MortonCodeType upper_bound)
            : left(left), right(right), lower_bound(lower_bound), upper_bound(upper_bound) {}
    };

    template <typename MortonCodeType = std::uint32_t>
    struct check_intersect_box_t {
        MortonCodeType query_lower_bound, query_upper_bound;

        explicit check_intersect_box_t(MortonCodeType query_lower_bound, MortonCodeType query_upper_bound)
            : query_lower_bound(query_lower_bound), query_upper_bound(query_upper_bound) {}

        bool operator()(const MortonCodeType& lower_bound, const MortonCodeType& upper_bound, 
            MortonCodeType x_mask, MortonCodeType y_mask, MortonCodeType z_mask) const 
        {
            return mc_is_less_than_or_equal(query_lower_bound, upper_bound, x_mask, y_mask, z_mask) && 
                   mc_is_less_than_or_equal(lower_bound, query_upper_bound, x_mask, y_mask, z_mask);
        }

        bool operator()(const MortonCodeType& morton_code, 
            MortonCodeType x_mask, MortonCodeType y_mask, MortonCodeType z_mask) const 
        {
            return operator()(morton_code, morton_code, x_mask, y_mask, z_mask);
        }
    };

    template <
        typename MortonCodeType = std::uint32_t, 
        typename IndexType = std::int32_t,
        template<typename, typename...> class Container = vec
    >
    class binary_radix_tree_t {
        using inode_t = internal_node_t<MortonCodeType, IndexType>;
        using morton_container_t = Container<MortonCodeType>;
        using node_container_t = Container<inode_t>;
        using result_t = Container<IndexType>;
    
    private: 
        node_container_t internal_nodes;
        morton_container_t leaf_nodes;
        MortonCodeType x_mask, y_mask, z_mask;
        
        IndexType count_leading_zeros(MortonCodeType x, IndexType num_bits = 30) const 
        {        
            return num_bits - (sizeof(MortonCodeType) * 8 - __builtin_clz(x));
        }

        void create_masks() 
        {
            for (int i = 0; i < sizeof(MortonCodeType) * 8; i += 3) {
                x_mask |= static_cast<MortonCodeType>(1) << i;
                y_mask |= static_cast<MortonCodeType>(1) << (i + 1);
                z_mask |= static_cast<MortonCodeType>(1) << (i + 2);
            }
        }

    public: 
        binary_radix_tree_t() {};

        binary_radix_tree_t(const morton_container_t& morton_codes, const IndexType num_bits = 30) 
        {
            build_tree(morton_codes, num_bits);
        }

        const node_container_t& get_internal_nodes() const 
        {
            return internal_nodes;
        }

        const morton_container_t& get_leaf_nodes() const 
        {
            return leaf_nodes;
        }

        MortonCodeType get_x_mask() const { return x_mask; }
        MortonCodeType get_y_mask() const { return y_mask; }
        MortonCodeType get_z_mask() const { return z_mask; }
        
        IndexType delta(const morton_container_t& morton_codes, IndexType i, IndexType j, IndexType num_bits = 30) const 
        {
            assert(i >= 0 && i < morton_codes.size());
            if (j < 0 || j >= morton_codes.size()) {
                return static_cast<IndexType>(-1);
            }
            return static_cast<IndexType>(count_leading_zeros(morton_codes[i] ^ morton_codes[j], num_bits));
        }

        void determine_range(const morton_container_t& morton_codes, IndexType i, IndexType& first, IndexType& last, IndexType num_bits = 30) const 
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

        IndexType find_split(const morton_container_t& morton_codes, IndexType delta_node, IndexType first, IndexType last, IndexType num_bits = 30) const 
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

        void find_lower_upper_bounds(MortonCodeType prefix, IndexType num_bits_prefix, MortonCodeType& lower_bound, MortonCodeType& upper_bound, IndexType num_bits = 30) const 
        {
            MortonCodeType mask1 = ((1 << num_bits_prefix) - 1) << (num_bits - num_bits_prefix);
            MortonCodeType mask2 = mask1 ^ ((1 << num_bits) - 1);

            lower_bound = prefix & mask1;
            upper_bound = prefix | mask2;
        }
        
        void build_tree(const morton_container_t& morton_codes, const IndexType num_bits = 30) 
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
            // create masks 
            create_masks();
        }

        template <typename OpType> 
        result_t traverse(OpType &check_intersect) const 
        {
            result_t result;
            stack<IndexType> s;
            auto n = leaf_nodes.size();
            s.push(n);
            while (!s.empty()) {
                IndexType node_index = s.top();
                s.pop();
                
                if (node_index < n) // leaf node 
                {
                    if (check_intersect(leaf_nodes[node_index], x_mask, y_mask, z_mask)) {
                        result.push_back(node_index);
                    }
                } else { // internal node
                    node_index -= n;
                    if (check_intersect(internal_nodes[node_index].lower_bound, internal_nodes[node_index].upper_bound, 
                        x_mask, y_mask, z_mask)) 
                    {
                        s.push(internal_nodes[node_index].left);
                        s.push(internal_nodes[node_index].right);
                    }
                }
            }
            std::sort(result.begin(), result.end());
            return result;
        }

    };

}}
