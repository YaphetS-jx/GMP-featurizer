#pragma once

#include <bitset>
#include <iostream>

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
    void bit_deinterleave_bits(BinaryType morton_code, BitSizeType num_bits_per_dim, BinaryType& x, BinaryType& y, BinaryType& z) 
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
    CodeType coordinate_to_morton_code(FractionalType num, BitSizeType num_bits) 
    {
        assert(num >= 0 && num < 1.0);        
        return fractional_to_binary<FractionalType, BitSizeType, CodeType>(num, num_bits - 1);
    }

    template <typename BinaryType, typename NumBitType, typename FractionalType>
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
    FractionalType morton_code_to_coordinate(CodeType morton_code, BitSizeType num_bits) 
    {
        assert(num_bits >= 1 && 3 * num_bits <= sizeof(CodeType) * 8 && (morton_code & (static_cast<CodeType>(1) << (num_bits - 1))) == 0);
        return binary_to_fractional<CodeType, BitSizeType, FractionalType>(morton_code, num_bits - 1);
    }

    // compare 2 morton codes in 3 different dimensions
    template <typename BinaryType>
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
}}}