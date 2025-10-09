#pragma once
#include <cuda_runtime.h>
#include "containers.hpp"
#include "math.hpp"
#include "geometry.hpp"

namespace gmp { namespace util {

    using namespace gmp::math;
    using gmp::geometry::point3d_t;
    using gmp::containers::vector;

    template <typename DataType, typename IndexType>
    __inline__ __host__ __device__
    IndexType binary_search(const DataType* data, IndexType start, IndexType end, const DataType elem)
    {
        while (start <= end)
        {
            IndexType mid_idx = (start + end) / 2;
            if (mid_idx == start && data[mid_idx] == elem) return mid_idx;
            else
            {
            if (mid_idx != start && data[mid_idx] == elem && data[mid_idx - 1] != elem) return mid_idx;
            else
            {
                if (data[mid_idx] < elem) start = mid_idx + 1;
                else end = mid_idx - 1;
            }
            }
        }

        return static_cast<IndexType>(-1); // nothing is found
    }


    template <typename DataType, typename IndexType>
    __inline__ __host__ __device__
    IndexType binary_search_first_larger(const DataType* data, IndexType start, IndexType end, const DataType target)
    {
        IndexType ans = end + 1;
        while (start <= end)
        {
            IndexType mid_idx = start + (end - start) / 2; // Avoid potential overflow
            // Move to right side if target is greater.
            if (data[mid_idx] <= target) {
                start = mid_idx + 1;
            } else {
            // Move left side.
                ans = mid_idx;
                if (mid_idx == 0) break; // Prevent underflow for unsigned types
                end = mid_idx - 1;
            }
        }
        return ans;
    }
    

    template <typename T>
    __device__
    inline T cuda_calculate_distance_squared(const sym_matrix3d_t<T> metric, const array3d_t<T> difference)
    {
        return difference.dot(metric * difference);
    }

    template <typename T>
    __device__
    inline T cuda_calculate_distance_squared(const sym_matrix3d_t<T> metric, const point3d_t<T> p1, const point3d_t<T> p2, 
        const array3d_t<T>& cell_shift, array3d_t<T>& difference) 
    {
        difference = array3d_t<T>{p1.x - p2.x, p1.y - p2.y, p1.z - p2.z};
        difference += cell_shift;
        return difference.dot(metric * difference);
    }

    template <typename T>
    __device__
    inline array3d_t<T> cuda_fractional_to_cartesian(const matrix3d_t<T> lattice_vectors, const array3d_t<T> fractional)
    {
        return lattice_vectors.transpose_mult(fractional);
    }

    
    template <typename T>
    void cuda_debug_print(const T* data, const int size, const int row_size,cudaStream_t stream)
    {
        vector<T> h_data(size);
        cudaMemcpyAsync(h_data.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (auto i = 0; i < size; i++) {
            std::cout << h_data[i] << ", ";
            if ((i + 1) % row_size == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    template <typename FloatType>
    __device__
    FloatType get_difference(FloatType min, FloatType max, FloatType point) {
        return (min <= point && point <= max) ? 0 : (point < min) ? min - point : point - max;
    };

}} // namespace gmp::util