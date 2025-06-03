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

    template <typename T>
    struct internal_node_t {    
        T index;
        T first, last;
        T split;
        T left, right;
        internal_node_t(T idx, T first, T last, T split, T left, T right)
            : index(idx), first(first), last(last), split(split), left(left), right(right) {}
    };

    template <
        typename MortonCodeType = std::uint32_t, 
        typename IndexType = std::int32_t,
        template<typename, typename...> class Container = vec
    >
    class binary_radix_tree_t {
    public: 
        using inode_t = internal_node_t<IndexType>;
        using morton_container_t = Container<MortonCodeType>;
        using node_container_t = Container<inode_t>;
        
        node_container_t internal_nodes;
        
        IndexType count_leading_zeros(MortonCodeType x) const {        
            return static_cast<IndexType>(std::__countl_zero(x));
        }

        IndexType delta(const morton_container_t& morton_codes, IndexType i, IndexType j) const {
            IndexType n = static_cast<IndexType>(morton_codes.size());
            if (i < 0 || i >= n || j < 0 || j >= n) {
                return static_cast<IndexType>(-1);
            }
            if (morton_codes[i] == morton_codes[j]) {
                // Same code -> fallback on index difference
                return static_cast<IndexType>(count_leading_zeros(static_cast<MortonCodeType>(i ^ j)) + 32);
            } else {                
                return static_cast<IndexType>(count_leading_zeros(morton_codes[i] ^ morton_codes[j]));
            }
        }

        void determine_range(const morton_container_t& morton_codes, IndexType i, IndexType& first, IndexType& last) const {        
            
            // Calculate direction based on delta differences         
            IndexType delta_prev = delta(morton_codes, i, i - 1);
            IndexType delta_next = delta(morton_codes, i, i + 1);
            IndexType d = (delta_next > delta_prev) ? 1 : -1;
            IndexType delta_min = delta(morton_codes, i, i - d);

            // Exponential search to find other end
            IndexType lmax = 2;
            while (delta(morton_codes, i, i + lmax * d) > delta_min) {            
                lmax *= 2;
            }

            // Binary search to refine
            IndexType l = 0;
            IndexType t = lmax / 2;
            while (t >= 1) {
                IndexType next = i + (l + t) * d;
                if (delta(morton_codes, i, next) > delta_min) {
                    l += t;
                }
                t /= 2;
            }

            IndexType j = i + l * d;
            first = std::min(i, j);
            last = std::max(i, j);
        }

        IndexType find_split(const morton_container_t& morton_codes, IndexType first, IndexType last) const {
            // Find the position where the prefix changes
            IndexType delta_node = delta(morton_codes, first, last);
            
            // Binary search for the split point
            IndexType split = first;
            IndexType stride = last - first;
            
            while (stride > 1) {
                stride = (stride + 1) / 2;
                IndexType mid = split + stride;
                
                if (mid < last) {
                    IndexType delta_mid = delta(morton_codes, first, mid);
                    if (delta_mid > delta_node) {
                        split = mid;
                    }
                }
            }
            return split;
        }
        
        void build_tree(const morton_container_t& morton_codes) {
            auto n = static_cast<IndexType>(morton_codes.size());
            internal_nodes.reserve(n - 1);

            for (IndexType i = 0; i < n - 1; ++i) {
                IndexType first, last;
                determine_range(morton_codes, i, first, last);
                IndexType split = find_split(morton_codes, first, last);

                // Determine left and right children
                IndexType left = (split == first) ? split : split + n;
                IndexType right = (split + 1 == last) ? split + 1 : split + 1 + n;

                internal_nodes.emplace_back(
                    i + n, first, last, split, left, right
                );
            }
        }
    };

}}
