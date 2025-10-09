#pragma once
#include <cmath>
#include <type_traits>
#include <limits>
#include <functional>
#include <cassert>
#include "error.hpp"
#include "containers.hpp"
#include "gmp_float.hpp"
#include "gpu_qualifiers.hpp"

namespace gmp { namespace math {

    // check if a value is close to zero
    template <typename T>
    GPU_HOST_DEVICE bool isZero(T value) {
    #ifdef __CUDA_ARCH__
        // Device code: use hardcoded epsilon values
        if constexpr (std::is_same_v<T, float>) {
            return std::abs(value) < 1.19209e-07f;  // FLT_EPSILON
        } else if constexpr (std::is_same_v<T, double>) {
            return std::abs(value) < 2.22045e-16;   // DBL_EPSILON  
        } else {
            return std::abs(value) < T(1e-14);      // Generic fallback
        }
    #else
        // Host code: use std::numeric_limits
        return std::abs(value) < std::numeric_limits<T>::epsilon();
    #endif
    }

    // check if two values are equal
    template<typename T>
    GPU_HOST_DEVICE typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b) 
    {
        return a == b; // exact comparison for non-floating-point types
    };

    // Floating-point specialization with relative+absolute tolerance
    template<typename T>
    GPU_HOST_DEVICE typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b,
            T absTol = static_cast<T>(1e-14),
            T relTol = static_cast<T>(1e-14)) 
    {
        return std::fabs(a - b) <= std::max(absTol, relTol * std::max(std::fabs(a), std::fabs(b)));
    }

    template <typename T>
    struct alignas(sizeof(T) * 2) array_2d_t {
        T data_[2];

        // Array access
        GPU_HOST_DEVICE T& operator[](size_t i) { 
            assert(i < 2); 
            return data_[i]; 
        }
        GPU_HOST_DEVICE const T& operator[](size_t i) const { 
            assert(i < 2); 
            return data_[i]; 
        }
    };

    template <typename T>
    struct alignas(sizeof(T) * 4) array3d_t {
        T data_[3];
        T padding_;

        // extra data - template version for different integer types
        template<typename IntType>
        GPU_HOST_DEVICE void set_extra(IntType v) {
            static_assert(std::is_integral_v<IntType>, "IntType must be an integral type");
            if constexpr (sizeof(T) == 4) {
                if constexpr (sizeof(IntType) <= 4) {
                    std::uint32_t val = static_cast<std::uint32_t>(v);
                    padding_ = *reinterpret_cast<const T*>(&val);
                } else {
                    // For larger int types, truncate to 32 bits
                    std::uint32_t val = static_cast<std::uint32_t>(v & 0xFFFFFFFFu);
                    padding_ = *reinterpret_cast<const T*>(&val);
                }
            } else { // sizeof(T) == 8
                if constexpr (sizeof(IntType) <= 8) {
                    std::uint64_t val = static_cast<std::uint64_t>(v);
                    padding_ = *reinterpret_cast<const T*>(&val);
                } else {
                    // For larger int types, truncate to 64 bits
                    std::uint64_t val = static_cast<std::uint64_t>(v & 0xFFFFFFFFFFFFFFFFu);
                    padding_ = *reinterpret_cast<const T*>(&val);
                }
            }
        }
        
        template<typename IntType>
        GPU_HOST_DEVICE IntType get_extra() const {
            static_assert(std::is_integral_v<IntType>, "IntType must be an integral type");
            if constexpr (sizeof(T) == 4) {
                auto val = *reinterpret_cast<const std::uint32_t*>(&padding_);
                if constexpr (std::is_signed_v<IntType>) {
                    return static_cast<IntType>(static_cast<std::int32_t>(val));
                } else {
                    return static_cast<IntType>(val);
                }
            } else { // sizeof(T) == 8
                auto val = *reinterpret_cast<const std::uint64_t*>(&padding_);
                if constexpr (std::is_signed_v<IntType>) {
                    return static_cast<IntType>(static_cast<std::int64_t>(val));
                } else {
                    return static_cast<IntType>(val);
                }
            }
        }
        
        // Convenience methods for backward compatibility
        GPU_HOST_DEVICE void set_extra(int v) {
            set_extra<int>(v);
        }
        GPU_HOST_DEVICE int get_extra() const {
            return get_extra<int>();
        }

        // Array access
        GPU_HOST_DEVICE T& operator[](size_t i) { 
            return data_[i]; 
        }
        GPU_HOST_DEVICE const T& operator[](size_t i) const { 
            return data_[i]; 
        }

        // Basic arithmetic operations
        GPU_HOST_DEVICE array3d_t<T> operator+(const array3d_t<T>& other) const {
            return {data_[0] + other.data_[0], data_[1] + other.data_[1], data_[2] + other.data_[2]};
        }
        GPU_HOST_DEVICE array3d_t<T>& operator+=(const array3d_t<T>& other) {
            data_[0] += other.data_[0];
            data_[1] += other.data_[1];
            data_[2] += other.data_[2];
            return *this;
        }
        GPU_HOST_DEVICE array3d_t<T> operator-(const array3d_t<T>& other) const {
            return {data_[0] - other.data_[0], data_[1] - other.data_[1], data_[2] - other.data_[2]};
        }
        GPU_HOST_DEVICE array3d_t<T>& operator-=(const array3d_t<T>& other) {
            data_[0] -= other.data_[0];
            data_[1] -= other.data_[1];
            data_[2] -= other.data_[2];
            return *this;
        }
        GPU_HOST_DEVICE array3d_t<T> operator*(const T scalar) const {
            return {data_[0] * scalar, data_[1] * scalar, data_[2] * scalar};
        }
        GPU_HOST_DEVICE array3d_t<T> operator*(const array3d_t<T>& arr) const {
            return {data_[0] * arr[0], data_[1] * arr[1], data_[2] * arr[2]};
        }
        GPU_HOST_DEVICE array3d_t<T> operator/(const T scalar) const {
            return {data_[0] / scalar, data_[1] / scalar, data_[2] / scalar};
        }

        // comparison operator
        GPU_HOST_DEVICE bool operator==(const array3d_t<T>& other) const {
            return data_[0] == other.data_[0] && data_[1] == other.data_[1] && data_[2] == other.data_[2] && padding_ == other.padding_;
        }
        GPU_HOST_DEVICE bool operator!=(const array3d_t<T>& other) const {
            return !(*this == other);
        }

        // Dot product
        GPU_HOST_DEVICE T dot(const array3d_t<T>& other) const {
            return data_[0] * other.data_[0] + data_[1] * other.data_[1] + data_[2] * other.data_[2];
        }

        // Cross product
        GPU_HOST_DEVICE array3d_t<T> cross(const array3d_t<T>& other) const {
            return {
                data_[1] * other.data_[2] - data_[2] * other.data_[1],
                data_[2] * other.data_[0] - data_[0] * other.data_[2],
                data_[0] * other.data_[1] - data_[1] * other.data_[0]
            };
        }

        // norm
        GPU_HOST_DEVICE T norm() const {
            return std::sqrt(data_[0] * data_[0] + data_[1] * data_[1] + data_[2] * data_[2]);
        }

        // product
        GPU_HOST_DEVICE T prod() const {
            return data_[0] * data_[1] * data_[2];
        }        
    };

    // Standalone array3d_t operators
    template <typename T>
    GPU_HOST_DEVICE array3d_t<T> operator*(const T scalar, const array3d_t<T>& arr) {
        return arr * scalar;
    }

    template <typename T>
    GPU_HOST_DEVICE array3d_t<T> operator/(const T scalar, const array3d_t<T>& arr) {
        return {scalar / arr[0], scalar / arr[1], scalar / arr[2]};
    }

    template<typename T>
    GPU_HOST_DEVICE
    array3d_t<T> make_array3d(T x, T y, T z) {
        array3d_t<T> result;
        result[0] = x;
        result[1] = y;
        result[2] = z;
        return result;
    }

    // type aliases using configured floating-point type
    using array3d_flt = array3d_t<gmp::gmp_float>;
    using array3d_uint32 = array3d_t<uint32_t>;
    using array3d_int32 = array3d_t<int32_t>;
    using array3d_int8 = array3d_t<int8_t>;
    using array3d_bool = std::array<bool, 3>;

    template <typename T>
    struct matrix3d_t {
        // data
        array3d_t<T> data_[3];

        // matrix access
        GPU_HOST_DEVICE array3d_t<T>& operator[](size_t i) { 
            return data_[i]; 
        }
        GPU_HOST_DEVICE const array3d_t<T>& operator[](size_t i) const { 
            return data_[i]; 
        }

        // comparison operator
        GPU_HOST_DEVICE bool operator==(const matrix3d_t<T>& other) const {
            return data_[0] == other.data_[0] && data_[1] == other.data_[1] && data_[2] == other.data_[2];
        }
        GPU_HOST_DEVICE bool operator!=(const matrix3d_t<T>& other) const {
            return !(*this == other);
        }

        // matrix multiplication
        GPU_HOST_DEVICE matrix3d_t<T> operator*(const matrix3d_t<T>& other) const {
            matrix3d_t<T> result;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    T sum = 0;
                    for (int k = 0; k < 3; k++) {
                        sum += data_[i][k] * other[k][j];
                    }                    
                    result[i][j] = sum;
                }
            }
            return result;
        }
        
        // matrix vector multiplication
        GPU_HOST_DEVICE array3d_t<T> operator*(const array3d_t<T>& other) const {
            array3d_t<T> result;
            for (int i = 0; i < 3; i++) {
                result[i] = data_[i].dot(other);
            }
            return result;
        }

        // transpose matrix vector multiplication
        GPU_HOST_DEVICE array3d_t<T> transpose_mult(const array3d_t<T>& other) const {
            array3d_t<T> result;
            for (int i = 0; i < 3; i++) {
                result[i] = data_[0][i] * other[0] + data_[1][i] * other[1] + data_[2][i] * other[2];
            }
            return result;
        }

        // matrix transpose
        GPU_HOST_DEVICE matrix3d_t<T> transpose() const {
            matrix3d_t<T> result;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result[j][i] = data_[i][j];
                }
            }
            return result;
        }
        
        // matrix determinant
        GPU_HOST_DEVICE T det() const {
            return data_[0][0]*(data_[1][1]*data_[2][2] - data_[1][2]*data_[2][1])
                   - data_[0][1]*(data_[1][0]*data_[2][2] - data_[1][2]*data_[2][0])
                   + data_[0][2]*(data_[1][0]*data_[2][1] - data_[1][1]*data_[2][0]);
        }

        // matrix inverse - optimized version
        GPU_HOST_DEVICE matrix3d_t<T> inverse() const {
            // Calculate determinant directly
            T det = this->det();
            if (isZero(det)) {
                update_error(error_t::matrix_singular);
                return matrix3d_t<T>();  // Return zero matrix for singular case
            }
            
            // Calculate inverse in-place
            matrix3d_t<T> result;
            T inv_det = 1.0/det;  // Single division
            
            result[0][0] = (data_[1][1]*data_[2][2] - data_[1][2]*data_[2][1]) * inv_det;
            result[0][1] = (data_[0][2]*data_[2][1] - data_[0][1]*data_[2][2]) * inv_det;
            result[0][2] = (data_[0][1]*data_[1][2] - data_[0][2]*data_[1][1]) * inv_det;
            result[1][0] = (data_[1][2]*data_[2][0] - data_[1][0]*data_[2][2]) * inv_det;
            result[1][1] = (data_[0][0]*data_[2][2] - data_[0][2]*data_[2][0]) * inv_det;
            result[1][2] = (data_[0][2]*data_[1][0] - data_[0][0]*data_[1][2]) * inv_det;
            result[2][0] = (data_[1][0]*data_[2][1] - data_[1][1]*data_[2][0]) * inv_det;
            result[2][1] = (data_[0][1]*data_[2][0] - data_[0][0]*data_[2][1]) * inv_det;
            result[2][2] = (data_[0][0]*data_[1][1] - data_[0][1]*data_[1][0]) * inv_det;
            
            return result;
        }
    };

    template<typename T>
    GPU_HOST_DEVICE
    matrix3d_t<T> make_matrix3d(array3d_t<T> row0, array3d_t<T> row1, array3d_t<T> row2) {
        matrix3d_t<T> result;
        result[0] = row0;
        result[1] = row1;
        result[2] = row2;
        return result;
    }

    // type aliases using configured floating-point type
    using matrix3d_flt = matrix3d_t<gmp::gmp_float>;

    template <typename T>
    struct sym_matrix3d_t {
        // data
        array3d_t<T> diag_;
        array3d_t<T> off_diag_;

        // matrix access - returns diagonal elements for i=0,1,2 and off-diagonal for i=3,4,5
        GPU_HOST_DEVICE T& operator[](size_t i) { 
            if(i < 3) return diag_[i];
            return off_diag_[i-3];
        }
        GPU_HOST_DEVICE const T& operator[](size_t i) const { 
            if(i < 3) return diag_[i];
            return off_diag_[i-3];
        }

        // comparison operator
        GPU_HOST_DEVICE bool operator==(const sym_matrix3d_t<T>& other) const {
            return diag_ == other.diag_ && off_diag_ == other.off_diag_;
        }
        GPU_HOST_DEVICE bool operator!=(const sym_matrix3d_t<T>& other) const {
            return !(*this == other);
        }

        // matrix vector multiplication
        GPU_HOST_DEVICE array3d_t<T> operator*(const array3d_t<T>& vec) const {            
            array3d_t<T> result = { 
                diag_[0]*vec[0] + off_diag_[0]*vec[1] + off_diag_[1]*vec[2],
                off_diag_[0]*vec[0] + diag_[1]*vec[1] + off_diag_[2]*vec[2],
                off_diag_[1]*vec[0] + off_diag_[2]*vec[1] + diag_[2]*vec[2]
            };
            return result;
        }
    };

    // type aliases using configured floating-point type
    using sym_matrix3d_flt = sym_matrix3d_t<gmp::gmp_float>;

    // other math functions 
    template <typename T>
    GPU_HOST_DEVICE T round_to_0_1(const T x) 
    {
        // Use fmod to get the fractional part (value between -1 and 1)
        T result = std::fmod(x, 1.0);
        
        // If the result is negative, add 1 to make it between 0 and 1
        if (result < 0.0) {
            result += 1.0;
        }
        
        // Handle the edge case of -0.0 or exact 1.0
        if (result == 0.0 && std::signbit(result)) {
            result = 0.0;  // Convert -0.0 to +0.0
        } else if (result == 1.0) {
            result = 0.0;  // 1.0 should wrap to 0.0
        }
        
        return result;
    }

    // functions 
    template <typename T>
    GPU_HOST_DEVICE T weighted_square_sum(const int mcsh_order, const T* v) 
    {
        switch (mcsh_order) {
        case -1:
            return v[0] * v[0];
        case 0:
            return v[0] * v[0];            
        case 1:
            return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        case 2:
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) 
                    + (2.0 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5]));
        case 3:
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) 
                    + (3.0 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7] + v[8] * v[8]))
                    + (6.0 * v[9] * v[9]);
        case 4:
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) 
                    + (4.0 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7] + v[8] * v[8]))
                    + (6.0 * (v[9] * v[9] + v[10] * v[10] + v[11] * v[11]))
                    + (12.0 * (v[12] * v[12] + v[13] * v[13] + v[14] * v[14]));
        case 5:
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) 
                    + (5.0 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7] + v[8] * v[8]))
                    + (10.0 * (v[9] * v[9] + v[10] * v[10] + v[11] * v[11] + v[12] * v[12] + v[13] * v[13] + v[14] * v[14]))
                    + (20.0 * (v[15] * v[15] + v[16] * v[16] + v[17] * v[17]))
                    + (30.0 * (v[18] * v[18] + v[19] * v[19] + v[20] * v[20]));
        case 6:
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) 
                    + (6.0 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7] + v[8] * v[8]))
                    + (15.0 * (v[9] * v[9] + v[10] * v[10] + v[11] * v[11] + v[12] * v[12] + v[13] * v[13] + v[14] * v[14]))
                    + (30.0 * (v[15] * v[15] + v[16] * v[16] + v[17] * v[17]))
                    + (20.0 * (v[18] * v[18] + v[19] * v[19] + v[20] * v[20]))
                    + (60.0 * (v[21] * v[21] + v[22] * v[22] + v[23] * v[23] + v[24] * v[24] + v[25] * v[25] + v[26] * v[26]))
                    + (90.0 * v[27] * v[27]);
        case 7:
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) 
            + (7.0 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7] + v[8] * v[8]))
            + (21.0 * (v[9] * v[9] + v[10] * v[10] + v[11] * v[11] + v[12] * v[12] + v[13] * v[13] + v[14] * v[14]))
            + (42.0 * (v[15] * v[15] + v[16] * v[16] + v[17] * v[17]))
            + (35.0 * (v[18] * v[18] + v[19] * v[19] + v[20] * v[20] + v[21] * v[21] + v[22] * v[22] + v[23] * v[23]))
            + (105.0 * (v[24] * v[24] + v[25] * v[25] + v[26] * v[26] + v[27] * v[27] + v[28] * v[28] + v[29] * v[29]))
            + (140.0 * (v[30] * v[30] + v[31] * v[31] + v[32] * v[32]))
            + (210.0 * (v[33] * v[33] + v[34] * v[34] + v[35] * v[35]));
        case 8:
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) 
            + (8.0 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7] + v[8] * v[8]))
            + (28.0 * (v[9] * v[9] + v[10] * v[10] + v[11] * v[11] + v[12] * v[12] + v[13] * v[13] + v[14] * v[14]))
            + (56.0 * (v[15] * v[15] + v[16] * v[16] + v[17] * v[17]))
            + (56.0 * (v[18] * v[18] + v[19] * v[19] + v[20] * v[20] + v[21] * v[21] + v[22] * v[22] + v[23] * v[23]))
            + (168.0 * (v[24] * v[24] + v[25] * v[25] + v[26] * v[26] + v[27] * v[27] + v[28] * v[28] + v[29] * v[29]))
            + (70.0 * (v[30] * v[30] + v[31] * v[31] + v[32] * v[32]))
            + (280.0 * (v[33] * v[33] + v[34] * v[34] + v[35] * v[35] + v[36] * v[36] + v[37] * v[37] + v[38] * v[38]))
            + (420.0 * (v[39] * v[39] + v[40] * v[40] + v[41] * v[41]))
            + (560.0 * (v[42] * v[42] + v[43] * v[43] + v[44] * v[44]));
        case 9:
            return (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) 
            + (9.0 * (v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6] + v[7] * v[7] + v[8] * v[8]))
            + (36.0 * (v[9] * v[9] + v[10] * v[10] + v[11] * v[11] + v[12] * v[12] + v[13] * v[13] + v[14] * v[14]))
            + (72.0 * (v[15] * v[15] + v[16] * v[16] + v[17] * v[17]))
            + (84.0 * (v[18] * v[18] + v[19] * v[19] + v[20] * v[20] + v[21] * v[21] + v[22] * v[22] + v[23] * v[23]))
            + (252.0 * (v[24] * v[24] + v[25] * v[25] + v[26] * v[26] + v[27] * v[27] + v[28] * v[28] + v[29] * v[29]))
            + (126.0 * (v[30] * v[30] + v[31] * v[31] + v[32] * v[32] + v[33] * v[33] + v[34] * v[34] + v[35] * v[35]))
            + (504.0 * (v[36] * v[36] + v[37] * v[37] + v[38] * v[38] + v[39] * v[39] + v[40] * v[40] + v[41] * v[41]))
            + (756.0 * (v[42] * v[42] + v[43] * v[43] + v[44] * v[44]))
            + (630.0 * (v[45] * v[45] + v[46] * v[46] + v[47] * v[47]))
            + (1260.0 * (v[48] * v[48] + v[49] * v[49] + v[50] * v[50] + v[51] * v[51] + v[52] * v[52] + v[53] * v[53]))
            + (1680.0 * v[54] * v[54]);
        default:
            return 0.0;
        }
    }
}}