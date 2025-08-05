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

    // Type aliases for external structures
    using gmp::containers::vector;
    using gmp::containers::deque;

    // check if a value is close to zero
    template <typename T>
    GPU_HOST_DEVICE bool isZero(T value);

    // check if two values are equal
    template<typename T>
    GPU_HOST_DEVICE typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b);

    // Floating-point specialization with relative+absolute tolerance
    template<typename T>
    GPU_HOST_DEVICE typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b,
            T absTol = static_cast<T>(1e-14),
            T relTol = static_cast<T>(1e-14));

    template <typename T>
    struct array_2d_t {
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
    struct array3d_t {
        T data_[3];

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
            return data_[0] == other.data_[0] && data_[1] == other.data_[1] && data_[2] == other.data_[2];
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
    GPU_HOST_DEVICE T round_to_0_1(const T x);

    // functions 
    template <typename T>
    GPU_HOST_DEVICE T weighted_square_sum(const int mcsh_order, const vector<T>& v);
}}