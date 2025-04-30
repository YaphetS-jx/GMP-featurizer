#pragma once
#include <cmath>
#include <type_traits>
#include <limits>

namespace gmp { namespace math {

    // check if a value is close to zero
    template <typename T>
    bool isZero(T value) {
        return std::abs(value) < std::numeric_limits<T>::epsilon();
    }

    // check if two values are equal
    template<typename T>
    typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b) {
        return a == b; // exact comparison for non-floating-point types
    }

    // Floating-point specialization with relative+absolute tolerance
    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b,
            T absTol = static_cast<T>(1e-14),
            T relTol = static_cast<T>(1e-14)) {
        return std::fabs(a - b) <= std::max(absTol, relTol * std::max(std::fabs(a), std::fabs(b)));
    }

    template <typename T>
    class array3d_t {
    public: 
        // constructor
        array3d_t() : data{0, 0, 0} {}
        array3d_t(T x, T y, T z) : data{x, y, z} {}
        array3d_t(const array3d_t<T>& other) : data{other.data[0], other.data[1], other.data[2]} {}
        array3d_t(array3d_t<T>&& other) noexcept : data{other.data[0], other.data[1], other.data[2]} {}

        // assignment operator
        array3d_t<T>& operator=(const array3d_t<T>& other) {
            data[0] = other.data[0];
            data[1] = other.data[1];
            data[2] = other.data[2];
            return *this;
        }
        array3d_t<T>& operator=(array3d_t<T>&& other) noexcept {
            if (this != &other) {
                data[0] = other.data[0];
                data[1] = other.data[1];
                data[2] = other.data[2];
            }
            return *this;
        }

        // Array access
        T& operator[](size_t i) { return data[i]; }
        const T& operator[](size_t i) const { return data[i]; }

        // Basic arithmetic operations
        array3d_t<T> operator+(const array3d_t<T>& other) const {
            return {data[0] + other.data[0], data[1] + other.data[1], data[2] + other.data[2]};
        }

        array3d_t<T> operator-(const array3d_t<T>& other) const {
            return {data[0] - other.data[0], data[1] - other.data[1], data[2] - other.data[2]};
        }

        array3d_t<T> operator*(const T scalar) const {
            return {data[0] * scalar, data[1] * scalar, data[2] * scalar};
        }

        friend array3d_t<T> operator*(const T scalar, const array3d_t<T>& arr) {
            return arr * scalar;
        }

        array3d_t<T> operator/(const T scalar) const {
            return {data[0] / scalar, data[1] / scalar, data[2] / scalar};
        }

        // comparsion operator
        bool operator==(const array3d_t<T>& other) const {
            return isEqual(data[0], other.data[0]) && isEqual(data[1], other.data[1]) && isEqual(data[2], other.data[2]);
        }
        bool operator!=(const array3d_t<T>& other) const {
            return !(*this == other);
        }

        // Dot product
        T dot(const array3d_t<T>& other) const {
            return data[0] * other.data[0] + data[1] * other.data[1] + data[2] * other.data[2];
        }

        // Cross product
        array3d_t<T> cross(const array3d_t<T>& other) const {
            return {
                data[1] * other.data[2] - data[2] * other.data[1],
                data[2] * other.data[0] - data[0] * other.data[2],
                data[0] * other.data[1] - data[1] * other.data[0]
            };
        }

        // norm
        T norm() const {
            return std::sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
        }
    private:
        T data[3];
    };

    // type aliases
    using array3d_flt64 = array3d_t<double>;
    using array3d_bool = array3d_t<bool>;

    template <typename T>
    class matrix3d_t {
    public:
        // constructor
        matrix3d_t() : data{array3d_t<T>{}, array3d_t<T>{}, array3d_t<T>{}} {}
        
        matrix3d_t(const array3d_t<T>& row0, const array3d_t<T>& row1, const array3d_t<T>& row2) 
            : data{row0, row1, row2} {}
                
        matrix3d_t(const matrix3d_t<T>& other)
            : data{other.data[0], other.data[1], other.data[2]} {}
        matrix3d_t(matrix3d_t<T>&& other) noexcept 
            : data{other.data[0], other.data[1], other.data[2]} {}

        // assignment operator
        matrix3d_t<T>& operator=(const matrix3d_t<T>& other) {
            data[0] = other.data[0];
            data[1] = other.data[1];
            data[2] = other.data[2];
            return *this;
        }
        matrix3d_t<T>& operator=(matrix3d_t<T>&& other) noexcept {
            if (this != &other) {
                data[0] = other.data[0];
                data[1] = other.data[1];
                data[2] = other.data[2];
            }
            return *this;
        }

        // matrix access
        array3d_t<T>& operator[](size_t i) { return data[i]; }
        const array3d_t<T>& operator[](size_t i) const { return data[i]; }

        // comparsion operator
        bool operator==(const matrix3d_t<T>& other) const {
            return data[0] == other.data[0] && data[1] == other.data[1] && data[2] == other.data[2];
        }
        bool operator!=(const matrix3d_t<T>& other) const {
            return !(*this == other);
        }

        // matrix multiplication
        matrix3d_t<T> operator*(const matrix3d_t<T>& other) const {
            matrix3d_t<T> result;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    T sum = 0;
                    for (int k = 0; k < 3; k++) {
                        sum += data[i][k] * other[k][j];
                    }                    
                    result[i][j] = sum;
                }
            }
            return result;
        }
        
        // matrix vector multiplication
        array3d_t<T> operator*(const array3d_t<T>& other) const {
            array3d_t<T> result;
            for (int i = 0; i < 3; i++) {
                result[i] = data[i].dot(other);
            }
            return result;
        }

        // matrix transpose
        matrix3d_t<T> transpose() const {
            matrix3d_t<T> result;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result[j][i] = data[i][j];
                }
            }
            return result;
        }
        
        // matrix determinant
        T det() const {
            return data[0][0]*(data[1][1]*data[2][2] - data[1][2]*data[2][1])
                   - data[0][1]*(data[1][0]*data[2][2] - data[1][2]*data[2][0])
                   + data[0][2]*(data[1][0]*data[2][1] - data[1][1]*data[2][0]);
        }

        // matrix inverse - optimized version
        matrix3d_t<T> inverse() const {
            // Calculate determinant directly
            T det = this->det();
            if (isZero(det)) {
                return matrix3d_t<T>();  // Return zero matrix for singular case
            }
            
            // Calculate inverse in-place
            matrix3d_t<T> result;
            T inv_det = 1.0/det;  // Single division
            
            result[0][0] = (data[1][1]*data[2][2] - data[1][2]*data[2][1]) * inv_det;
            result[0][1] = (data[0][2]*data[2][1] - data[0][1]*data[2][2]) * inv_det;
            result[0][2] = (data[0][1]*data[1][2] - data[0][2]*data[1][1]) * inv_det;
            result[1][0] = (data[1][2]*data[2][0] - data[1][0]*data[2][2]) * inv_det;
            result[1][1] = (data[0][0]*data[2][2] - data[0][2]*data[2][0]) * inv_det;
            result[1][2] = (data[0][2]*data[1][0] - data[0][0]*data[1][2]) * inv_det;
            result[2][0] = (data[1][0]*data[2][1] - data[1][1]*data[2][0]) * inv_det;
            result[2][1] = (data[0][1]*data[2][0] - data[0][0]*data[2][1]) * inv_det;
            result[2][2] = (data[0][0]*data[1][1] - data[0][1]*data[1][0]) * inv_det;
            
            return result;
        }

    private:
        array3d_t<T> data[3];
    };

    // type aliases
    using matrix3d_flt64 = matrix3d_t<double>;

    template <typename T>
    class sym_matrix3d_t {
    public:
        // constructor
        sym_matrix3d_t() : diag{array3d_t<T>{}}, off_diag{array3d_t<T>{}} {}
        sym_matrix3d_t(T a, T b, T c, T d, T e, T f) : diag{array3d_t<T>{a, b, c}}, off_diag{array3d_t<T>{d, e, f}} {}
        sym_matrix3d_t(const array3d_t<T>& diag_, const array3d_t<T>& off_diag_) : diag{diag_}, off_diag{off_diag_} {}        
        sym_matrix3d_t(const sym_matrix3d_t<T>& other) : diag{other.diag}, off_diag{other.off_diag} {}
        sym_matrix3d_t(sym_matrix3d_t<T>&& other) noexcept : diag{std::move(other.diag)}, off_diag{std::move(other.off_diag)} {}
        
        // assignment operator
        sym_matrix3d_t<T>& operator=(const sym_matrix3d_t<T>& other) {
            diag = other.diag;
            off_diag = other.off_diag;
            return *this;
        }

        // matrix access - returns diagonal elements for i=0,1,2 and off-diagonal for i=3,4,5
        T& operator[](size_t i) { 
            if(i < 3) return diag[i];
            return off_diag[i-3];
        }
        const T& operator[](size_t i) const { 
            if(i < 3) return diag[i];
            return off_diag[i-3];
        }

        // comparsion operator
        bool operator==(const sym_matrix3d_t<T>& other) const {
            return diag == other.diag && off_diag == other.off_diag;
        }
        bool operator!=(const sym_matrix3d_t<T>& other) const {
            return !(*this == other);
        }

        // matrix vector multiplication
        array3d_t<T> operator*(const array3d_t<T>& vec) const {            
            array3d_t<T> result = { 
                diag[0]*vec[0] + off_diag[0]*vec[1] + off_diag[1]*vec[2],
                off_diag[0]*vec[0] + diag[1]*vec[1] + off_diag[2]*vec[2],
                off_diag[1]*vec[0] + off_diag[2]*vec[1] + diag[2]*vec[2]
            };
            return result;
        }

    private: 
        array3d_t<T> diag;
        array3d_t<T> off_diag;
    };

    // type aliases
    using sym_matrix3d_flt64 = sym_matrix3d_t<double>;
}}