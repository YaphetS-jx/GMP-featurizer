#pragma once

#include <bitset>
#include <iostream>
#include <cassert>
#include "gpu_qualifiers.hpp"

namespace gmp { namespace tree { namespace morton_codes {

    template <typename T, typename U>
    void print_bits(T num, U num_bits) 
    {
        std::bitset<sizeof(T) * 8> bits(num);
        for (size_t i = 0; i < num_bits; ++i) {
            std::cout << bits[num_bits - 1 - i];
        }
        std::cout << std::endl;
    }

    template <typename BinaryType, typename BitSizeType>
    GPU_HOST_DEVICE
    BinaryType interleave_bits(BinaryType x, BinaryType y, BinaryType z, BitSizeType num_bits) 
    {
        assert(num_bits >= 1 && num_bits <= sizeof(BinaryType) * 8);

        BinaryType result = 0;
        for (BitSizeType i = 0; i < num_bits; ++i) {
            result |= (static_cast<BinaryType>(x & (static_cast<BinaryType>(1) << i)) << (2 * i)) 
                | (static_cast<BinaryType>(y & (static_cast<BinaryType>(1) << i)) << (2 * i + 1)) 
                | (static_cast<BinaryType>(z & (static_cast<BinaryType>(1) << i)) << (2 * i + 2));
        }
        return result;
    }

    template <typename BinaryType, typename BitSizeType>
    GPU_HOST_DEVICE
    void deinterleave_bits(BinaryType morton_code, BitSizeType num_bits_per_dim, BinaryType& x, BinaryType& y, BinaryType& z) 
    {
        assert(num_bits_per_dim >= 1 && num_bits_per_dim <= sizeof(BinaryType) * 8);

        x = 0; y = 0; z = 0;
        for (BitSizeType i = 0; i < num_bits_per_dim; ++i) {
            x |= (static_cast<BinaryType>(morton_code & (static_cast<BinaryType>(1) << (3 * i))) >> (2 * i));
            y |= (static_cast<BinaryType>(morton_code & (static_cast<BinaryType>(1) << (3 * i + 1))) >> (2 * i + 1));
            z |= (static_cast<BinaryType>(morton_code & (static_cast<BinaryType>(1) << (3 * i + 2))) >> (2 * i + 2));
        }
    }

    template <typename FractionalType, typename NumBitType, typename BinaryType>
    GPU_HOST_DEVICE
    BinaryType fractional_to_binary(FractionalType num, const NumBitType num_bits) 
    {
        assert(num >= 0 && num < 1.0);
        BinaryType result = 0;
        NumBitType n = 0;

        while (num > 0 && n < num_bits) {
            num *= 2;
            if (num >= 1.0) {
                result |= static_cast<BinaryType>(1) << (num_bits - 1 - n);
                num -= 1.0;
            }
            n++;
        }
        return result;
    }


    // the first bit is always 0, given that level 0 is the whole space
    // this is for single dimension morton codes
    template <typename FractionalType, typename BitSizeType, typename CodeType>
    GPU_HOST_DEVICE
    CodeType coordinate_to_morton_code(FractionalType num, BitSizeType num_bits) 
    {
        assert(num >= 0 && num < 1.0);        
        return fractional_to_binary<FractionalType, BitSizeType, CodeType>(num, num_bits - 1);
    }

    template <typename BinaryType, typename NumBitType, typename FractionalType>
    GPU_HOST_DEVICE
    FractionalType binary_to_fractional(BinaryType binary, const NumBitType num_bits) 
    {
        assert(num_bits <= sizeof(BinaryType) * 8);
        FractionalType result = 0;
        for (NumBitType i = 0; i < num_bits; ++i) {
            result += (binary & static_cast<BinaryType>(1)) * 1.0 / (static_cast<BinaryType>(1) << (num_bits - i));
            binary >>= 1;
        }
        return result;
    }

    // this is for single dimension morton codes
    template <typename FractionalType, typename BitSizeType, typename CodeType>
    GPU_HOST_DEVICE
    FractionalType morton_code_to_coordinate(CodeType morton_code, BitSizeType num_bits) 
    {
        assert(num_bits >= 1 && 3 * num_bits <= sizeof(CodeType) * 8 && (morton_code & (static_cast<CodeType>(1) << (num_bits - 1))) == 0);
        return binary_to_fractional<CodeType, BitSizeType, FractionalType>(morton_code, num_bits - 1);
    }

    // compare 2 morton codes in 3 different dimensions
    template <typename BinaryType>
    GPU_HOST_DEVICE
    bool mc_is_less_than_or_equal(const BinaryType morton_code1, const BinaryType morton_code2, 
        const BinaryType x_mask, const BinaryType y_mask, const BinaryType z_mask) 
    {
        BinaryType x1 = morton_code1 & x_mask;
        BinaryType y1 = morton_code1 & y_mask;
        BinaryType z1 = morton_code1 & z_mask;
        BinaryType x2 = morton_code2 & x_mask;
        BinaryType y2 = morton_code2 & y_mask;
        BinaryType z2 = morton_code2 & z_mask;
        return x1 <= x2 && y1 <= y2 && z1 <= z2;
    }

    template <typename MortonCodeType>
    GPU_HOST_DEVICE
    void create_masks(MortonCodeType& x_mask, MortonCodeType& y_mask, MortonCodeType& z_mask)
    {
        x_mask = y_mask = z_mask = 0;
        for (auto i = 0; i < sizeof(MortonCodeType) * 8; i += 3) {
            x_mask |= static_cast<MortonCodeType>(1) << i;
            y_mask |= static_cast<MortonCodeType>(1) << (i + 1);
            z_mask |= static_cast<MortonCodeType>(1) << (i + 2);
        }
    }

    template <typename MortonCodeType, typename IndexType>
    GPU_HOST_DEVICE
    IndexType count_leading_zeros(MortonCodeType x, IndexType num_bits)
    {
        return num_bits - (sizeof(MortonCodeType) * 8 - __builtin_clz(x));
    }

    template <typename MortonCodeType, typename IndexType>
    GPU_HOST_DEVICE
    IndexType delta(const MortonCodeType* morton_codes, IndexType num_mc, IndexType i, IndexType j, IndexType num_bits = 30)
    {
        if (i < 0 || i >= num_mc) {
            return static_cast<IndexType>(-1);
        }
        if (j < 0 || j >= num_mc) {
            return static_cast<IndexType>(-1);
        }
        return count_leading_zeros(morton_codes[i] ^ morton_codes[j], num_bits);
    }

    template <typename MortonCodeType, typename IndexType>
    GPU_HOST_DEVICE
    void determine_range(const MortonCodeType* morton_codes, IndexType num_mc, IndexType i, IndexType& first, IndexType& last, IndexType num_bits = 30)
    {
        assert(i >= 0 && i < num_mc);
        // Calculate direction based on delta differences         
        IndexType delta_prev = delta(morton_codes, num_mc, i, i - 1, num_bits);
        IndexType delta_next = delta(morton_codes, num_mc, i, i + 1, num_bits);
        IndexType d = (delta_next > delta_prev) ? 1 : -1;
        IndexType delta_min = delta(morton_codes, num_mc, i, i - d, num_bits);            

        // Exponential search to find other end
        IndexType lmax = 2;
        while (delta(morton_codes, num_mc, i, i + lmax * d, num_bits) > delta_min) {
            lmax *= 2;
        }

        // Binary search to refine
        IndexType l = 0;
        IndexType t = lmax / 2;
        while (t >= 1) {
            IndexType next = i + (l + t) * d;
            if (delta(morton_codes, num_mc, i, next, num_bits) > delta_min) {
                l += t;
            }
            t /= 2;
        }

        IndexType j = i + l * d;
        first = (i < j) ? i : j;
        last = (i > j) ? i : j;
    }

    template <typename MortonCodeType, typename IndexType>
    GPU_HOST_DEVICE
    IndexType find_split(const MortonCodeType* morton_codes, IndexType num_mc, IndexType delta_node, IndexType first, IndexType last, IndexType num_bits = 30)
    {
        assert(first >= 0 && first < num_mc);
        assert(last >= 0 && last < num_mc);                         
        
        // Binary search for the split point
        IndexType split = first;
        IndexType stride = last - first;
        
        while (stride > 1) {
            stride = (stride + 1) / 2;
            IndexType mid = split + stride;
            
            if (mid < last && delta(morton_codes, num_mc, first, mid, num_bits) > delta_node) {
                split = mid;
            }
        }
        return split;
    }

    template <typename MortonCodeType, typename IndexType>
    GPU_HOST_DEVICE
    void find_lower_upper_bounds(MortonCodeType prefix, IndexType num_bits_prefix, MortonCodeType& lower_bound, MortonCodeType& upper_bound, IndexType num_bits = 30)
    {
        MortonCodeType mask1 = ((1 << num_bits_prefix) - 1) << (num_bits - num_bits_prefix);
        MortonCodeType mask2 = mask1 ^ ((1 << num_bits) - 1);

        lower_bound = prefix & mask1;
        upper_bound = prefix | mask2;
    }
}}}