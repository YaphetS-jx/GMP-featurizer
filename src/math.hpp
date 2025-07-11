#pragma once
#include <cmath>
#include <type_traits>
#include <limits>
#include <functional>
#include <cassert>
#include "error.hpp"
#include "types.hpp"
#include "gmp_float.hpp"

namespace gmp { namespace math {

    // Type aliases for external structures
    using gmp::containers::vector;
    using gmp::containers::deque;

    // check if a value is close to zero
    template <typename T>
    bool isZero(T value);

    // check if two values are equal
    template<typename T>
    typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b);

    // Floating-point specialization with relative+absolute tolerance
    template<typename T>
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b,
            T absTol = static_cast<T>(1e-14),
            T relTol = static_cast<T>(1e-14));

    template <typename T>
    class array_2d_t {
    private: 
        T data_[2];
    public: 
        array_2d_t();
        array_2d_t(T x, T y);
        array_2d_t(const array_2d_t<T>& other);
        array_2d_t(array_2d_t<T>&& other) noexcept;

        // assignment operator
        array_2d_t<T>& operator=(const array_2d_t<T>& other);
        array_2d_t<T>& operator=(array_2d_t<T>&& other) noexcept;

        // Array access
        T& operator[](size_t i);
        const T& operator[](size_t i) const;
    };

    template <typename T>
    class array3d_t {
    public: 
        // constructor
        array3d_t();
        array3d_t(T x, T y, T z);
        array3d_t(const array3d_t<T>& other);
        array3d_t(array3d_t<T>&& other) noexcept;

        // assignment operator
        array3d_t<T>& operator=(const array3d_t<T>& other);
        array3d_t<T>& operator=(array3d_t<T>&& other) noexcept;

        // Array access
        T& operator[](size_t i);
        const T& operator[](size_t i) const;

        // Basic arithmetic operations
        array3d_t<T> operator+(const array3d_t<T>& other) const;
        array3d_t<T>& operator+=(const array3d_t<T>& other);
        array3d_t<T> operator-(const array3d_t<T>& other) const;
        array3d_t<T>& operator-=(const array3d_t<T>& other);
        array3d_t<T> operator*(const T scalar) const;
        array3d_t<T> operator*(const array3d_t<T>& arr) const;
        array3d_t<T> operator/(const T scalar) const;

        // comparison operator
        bool operator==(const array3d_t<T>& other) const;
        bool operator!=(const array3d_t<T>& other) const;

        // Dot product
        T dot(const array3d_t<T>& other) const;

        // Cross product
        array3d_t<T> cross(const array3d_t<T>& other) const;

        // norm
        T norm() const;

        // product
        T prod() const;

    private:
        T data_[3];
    };

    template <typename T>
    array3d_t<T> operator*(const T scalar, const array3d_t<T>& arr);

    template <typename T>
    array3d_t<T> operator/(const T scalar, const array3d_t<T>& arr);

    // type aliases using configured floating-point type
    using array3d_flt = array3d_t<gmp::gmp_float>;
    using array3d_uint32 = array3d_t<uint32_t>;
    using array3d_int32 = array3d_t<int32_t>;
    using array3d_int8 = array3d_t<int8_t>;
    using array3d_bool = std::array<bool, 3>;

    template <typename T>
    class matrix3d_t {
    public:
        // constructor
        matrix3d_t();
        matrix3d_t(const array3d_t<T>& row0, const array3d_t<T>& row1, const array3d_t<T>& row2);
        matrix3d_t(const matrix3d_t<T>& other);
        matrix3d_t(matrix3d_t<T>&& other) noexcept;

        // assignment operator
        matrix3d_t<T>& operator=(const matrix3d_t<T>& other);
        matrix3d_t<T>& operator=(matrix3d_t<T>&& other) noexcept;

        // matrix access
        array3d_t<T>& operator[](size_t i);
        const array3d_t<T>& operator[](size_t i) const;

        // comparison operator
        bool operator==(const matrix3d_t<T>& other) const;
        bool operator!=(const matrix3d_t<T>& other) const;

        // matrix multiplication
        matrix3d_t<T> operator*(const matrix3d_t<T>& other) const;
        
        // matrix vector multiplication
        array3d_t<T> operator*(const array3d_t<T>& other) const;

        // transpose matrix vector multiplication
        array3d_t<T> transpose_mult(const array3d_t<T>& other) const;

        // matrix transpose
        matrix3d_t<T> transpose() const;
        
        // matrix determinant
        T det() const;

        // matrix inverse - optimized version
        matrix3d_t<T> inverse() const;

    private:
        array3d_t<T> data_[3];
    };

    // type aliases using configured floating-point type
    using matrix3d_flt = matrix3d_t<gmp::gmp_float>;

    template <typename T>
    class sym_matrix3d_t {
    public:
        // constructor
        sym_matrix3d_t();
        sym_matrix3d_t(T a, T b, T c, T d, T e, T f);
        sym_matrix3d_t(const array3d_t<T>& diag, const array3d_t<T>& off_diag);
        sym_matrix3d_t(const sym_matrix3d_t<T>& other);
        sym_matrix3d_t(sym_matrix3d_t<T>&& other) noexcept;
        
        // assignment operator
        sym_matrix3d_t<T>& operator=(const sym_matrix3d_t<T>& other);

        // matrix access - returns diagonal elements for i=0,1,2 and off-diagonal for i=3,4,5
        T& operator[](size_t i);
        const T& operator[](size_t i) const;

        // comparison operator
        bool operator==(const sym_matrix3d_t<T>& other) const;
        bool operator!=(const sym_matrix3d_t<T>& other) const;

        // matrix vector multiplication
        array3d_t<T> operator*(const array3d_t<T>& vec) const;

    private: 
        array3d_t<T> diag_;
        array3d_t<T> off_diag_;
    };

    // type aliases using configured floating-point type
    using sym_matrix3d_flt = sym_matrix3d_t<gmp::gmp_float>;

    // other math functions 
    template <typename T>
    T round_to_0_1(const T x);

    // functions 
    template <typename T>
    T weighted_square_sum(const int mcsh_order, const vector<T>& v);
}}