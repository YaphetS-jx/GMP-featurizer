#pragma once
#include <xsimd/xsimd.hpp>
#include "types.hpp"

namespace gmp { namespace simd {

    template<typename T>
    class simd_info {
    private:
        simd_info() : batch_size_(xsimd::batch<T>::size) {}
        simd_info(const simd_info&) = delete;
        simd_info& operator=(const simd_info&) = delete;

        const size_t batch_size_;

    public:
        static simd_info& instance() {
            static simd_info singleton;
            return singleton;
        }

        size_t batch_size() const { return batch_size_; }
        using batch_type = xsimd::batch<T>;
    };

    template <typename T>
    using vector_aligned = gmp::containers::vector<T, xsimd::aligned_allocator<T>>;

    template<class T, size_t N>
    using batch_array = std::array<xsimd::batch<T>, N>;

    template<class T, size_t N>
    using aligned_array = std::array<vector_aligned<T>, N>;
    
    template <typename T>
    struct Params
    {
        vector_aligned<T> dx, dy, dz;
        vector_aligned<T> r_sqr;
        vector_aligned<T> C1, C2;
        vector_aligned<T> lambda, gamma;
        size_t num_elements;
    };

    template <typename T>
    struct Params_batch
    {
        using batch = xsimd::batch<T>;
        batch dx, dy, dz;
        batch r_sqr;
        batch C1, C2;
        batch lambda, gamma;
    };
}}
