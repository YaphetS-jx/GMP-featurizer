#pragma once
#include <cmath>
#include <type_traits>
#include <limits>
#include <functional>
#include "error.hpp"
#include "types.hpp"

namespace gmp { namespace math {

    using namespace gmp::containers;

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
        array3d_t() : data_{0, 0, 0} {}
        array3d_t(T x, T y, T z) : data_{x, y, z} {}
        array3d_t(const array3d_t<T>& other) : data_{other.data_[0], other.data_[1], other.data_[2]} {}
        array3d_t(array3d_t<T>&& other) noexcept : data_{other.data_[0], other.data_[1], other.data_[2]} {}

        // assignment operator
        array3d_t<T>& operator=(const array3d_t<T>& other) {
            data_[0] = other.data_[0];
            data_[1] = other.data_[1];
            data_[2] = other.data_[2];
            return *this;
        }
        array3d_t<T>& operator=(array3d_t<T>&& other) noexcept {
            if (this != &other) {
                data_[0] = other.data_[0];
                data_[1] = other.data_[1];
                data_[2] = other.data_[2];
            }
            return *this;
        }

        // Array access
        T& operator[](size_t i) { return data_[i]; }
        const T& operator[](size_t i) const { return data_[i]; }

        // Basic arithmetic operations
        array3d_t<T> operator+(const array3d_t<T>& other) const {
            return {data_[0] + other.data_[0], data_[1] + other.data_[1], data_[2] + other.data_[2]};
        }

        array3d_t<T>& operator+=(const array3d_t<T>& other) {
            data_[0] += other.data_[0];
            data_[1] += other.data_[1];
            data_[2] += other.data_[2];
            return *this;
        }

        array3d_t<T> operator-(const array3d_t<T>& other) const {
            return {data_[0] - other.data_[0], data_[1] - other.data_[1], data_[2] - other.data_[2]};
        }

        array3d_t<T>& operator-=(const array3d_t<T>& other) {
            data_[0] -= other.data_[0];
            data_[1] -= other.data_[1];
            data_[2] -= other.data_[2];
            return *this;
        }

        array3d_t<T> operator*(const T scalar) const {
            return {data_[0] * scalar, data_[1] * scalar, data_[2] * scalar};
        }
        
        array3d_t<T> operator*(const array3d_t<T>& arr) const {
            return {data_[0] * arr[0], data_[1] * arr[1], data_[2] * arr[2]};
        }

        friend array3d_t<T> operator*(const T scalar, const array3d_t<T>& arr) {
            return arr * scalar;
        }

        array3d_t<T> operator/(const T scalar) const {
            return {data_[0] / scalar, data_[1] / scalar, data_[2] / scalar};
        }

        // comparsion operator
        bool operator==(const array3d_t<T>& other) const {
            return isEqual(data_[0], other.data_[0]) && isEqual(data_[1], other.data_[1]) && isEqual(data_[2], other.data_[2]);
        }
        bool operator!=(const array3d_t<T>& other) const {
            return !(*this == other);
        }

        // Dot product
        T dot(const array3d_t<T>& other) const {
            return data_[0] * other.data_[0] + data_[1] * other.data_[1] + data_[2] * other.data_[2];
        }

        // Cross product
        array3d_t<T> cross(const array3d_t<T>& other) const {
            return {
                data_[1] * other.data_[2] - data_[2] * other.data_[1],
                data_[2] * other.data_[0] - data_[0] * other.data_[2],
                data_[0] * other.data_[1] - data_[1] * other.data_[0]
            };
        }

        // norm
        T norm() const {
            return std::sqrt(data_[0] * data_[0] + data_[1] * data_[1] + data_[2] * data_[2]);
        }

        // product
        T prod() const {
            return data_[0] * data_[1] * data_[2];
        }
    private:
        T data_[3];
    };

    // type aliases
    using array3d_flt64 = array3d_t<double>;
    using array3d_int32 = array3d_t<int32_t>;
    using array3d_int8 = array3d_t<int8_t>;
    using array3d_bool = array3d_t<bool>;

    template <typename T>
    class matrix3d_t {
    public:
        // constructor
        matrix3d_t() : data_{array3d_t<T>{}, array3d_t<T>{}, array3d_t<T>{}} {}
        
        matrix3d_t(const array3d_t<T>& row0, const array3d_t<T>& row1, const array3d_t<T>& row2) 
            : data_{row0, row1, row2} {}
                
        matrix3d_t(const matrix3d_t<T>& other)
            : data_{other.data_[0], other.data_[1], other.data_[2]} {}
        matrix3d_t(matrix3d_t<T>&& other) noexcept 
            : data_{other.data_[0], other.data_[1], other.data_[2]} {}

        // assignment operator
        matrix3d_t<T>& operator=(const matrix3d_t<T>& other) {
            data_[0] = other.data_[0];
            data_[1] = other.data_[1];
            data_[2] = other.data_[2];
            return *this;
        }
        matrix3d_t<T>& operator=(matrix3d_t<T>&& other) noexcept {
            if (this != &other) {
                data_[0] = other.data_[0];
                data_[1] = other.data_[1];
                data_[2] = other.data_[2];
            }
            return *this;
        }

        // matrix access
        array3d_t<T>& operator[](size_t i) { return data_[i]; }
        const array3d_t<T>& operator[](size_t i) const { return data_[i]; }

        // comparsion operator
        bool operator==(const matrix3d_t<T>& other) const {
            return data_[0] == other.data_[0] && data_[1] == other.data_[1] && data_[2] == other.data_[2];
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
                        sum += data_[i][k] * other[k][j];
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
                result[i] = data_[i].dot(other);
            }
            return result;
        }

        // matrix transpose
        matrix3d_t<T> transpose() const {
            matrix3d_t<T> result;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result[j][i] = data_[i][j];
                }
            }
            return result;
        }
        
        // matrix determinant
        T det() const {
            return data_[0][0]*(data_[1][1]*data_[2][2] - data_[1][2]*data_[2][1])
                   - data_[0][1]*(data_[1][0]*data_[2][2] - data_[1][2]*data_[2][0])
                   + data_[0][2]*(data_[1][0]*data_[2][1] - data_[1][1]*data_[2][0]);
        }

        // matrix inverse - optimized version
        matrix3d_t<T> inverse() const {
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

    private:
        array3d_t<T> data_[3];
    };

    // type aliases
    using matrix3d_flt64 = matrix3d_t<double>;

    template <typename T>
    class sym_matrix3d_t {
    public:
        // constructor
        sym_matrix3d_t() : diag_{array3d_t<T>{}}, off_diag_{array3d_t<T>{}} {}
        sym_matrix3d_t(T a, T b, T c, T d, T e, T f) : diag_{array3d_t<T>{a, b, c}}, off_diag_{array3d_t<T>{d, e, f}} {}
        sym_matrix3d_t(const array3d_t<T>& diag, const array3d_t<T>& off_diag) : diag_{diag}, off_diag_{off_diag} {}        
        sym_matrix3d_t(const sym_matrix3d_t<T>& other) : diag_{other.diag_}, off_diag_{other.off_diag_} {}
        sym_matrix3d_t(sym_matrix3d_t<T>&& other) noexcept : diag_{std::move(other.diag_)}, off_diag_{std::move(other.off_diag_)} {}
        
        // assignment operator
        sym_matrix3d_t<T>& operator=(const sym_matrix3d_t<T>& other) {
            diag_ = other.diag_;
            off_diag_ = other.off_diag_;
            return *this;
        }

        // matrix access - returns diagonal elements for i=0,1,2 and off-diagonal for i=3,4,5
        T& operator[](size_t i) { 
            if(i < 3) return diag_[i];
            return off_diag_[i-3];
        }
        const T& operator[](size_t i) const { 
            if(i < 3) return diag_[i];
            return off_diag_[i-3];
        }

        // comparsion operator
        bool operator==(const sym_matrix3d_t<T>& other) const {
            return diag_ == other.diag_ && off_diag_ == other.off_diag_;
        }
        bool operator!=(const sym_matrix3d_t<T>& other) const {
            return !(*this == other);
        }

        // matrix vector multiplication
        array3d_t<T> operator*(const array3d_t<T>& vec) const {            
            array3d_t<T> result = { 
                diag_[0]*vec[0] + off_diag_[0]*vec[1] + off_diag_[1]*vec[2],
                off_diag_[0]*vec[0] + diag_[1]*vec[1] + off_diag_[2]*vec[2],
                off_diag_[1]*vec[0] + off_diag_[2]*vec[1] + diag_[2]*vec[2]
            };
            return result;
        }

    private: 
        array3d_t<T> diag_;
        array3d_t<T> off_diag_;
    };

    // type aliases
    using sym_matrix3d_flt64 = sym_matrix3d_t<double>;

    // other math functions 
    inline double round_to_0_1(const double x) {
        // Use fmod to get the fractional part (value between -1 and 1)
        double result = std::fmod(x, 1.0);
        
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

    // solid mcsh functions
    void calculate_solid_mcsh_n1(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_0(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_1(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_2(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_3(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_4(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_5(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_6(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_7(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_8(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    void calculate_solid_mcsh_9(const array3d_flt64&, const double, const double, const double, const double, vec<double>&);
    
    // solid mcsh function registry
    class mcsh_function_registry_t {
        using solid_gmp_function_t = std::function<void(const array3d_flt64&, const double, const double, const double, const double, vec<double>&)>;
    public: 
        // Add a flag to control whether register_functions is called in constructor
        mcsh_function_registry_t(bool register_on_init = true) : functions_{}, num_values_{} {
            if (register_on_init) {
                register_functions();
            }
        }

        static mcsh_function_registry_t& get_instance() {
            static mcsh_function_registry_t instance;
            return instance;
        }

        const solid_gmp_function_t& get_function(int order) const {
            return order < 0 ? functions_[0] : functions_[order];
        }

        int get_num_values(int order) const {
            return order < 0 ? num_values_[0] : num_values_[order];
        }

        void register_functions() {
            // Create vectors with push_back instead of initializer lists to avoid memory issues
            
            // Reserve space first to avoid reallocations
            functions_.reserve(10);
            
            // Add functions one by one
            // functions_.push_back(calculate_solid_mcsh_n1);
            functions_.push_back(calculate_solid_mcsh_0);
            functions_.push_back(calculate_solid_mcsh_1);
            functions_.push_back(calculate_solid_mcsh_2);
            functions_.push_back(calculate_solid_mcsh_3);
            functions_.push_back(calculate_solid_mcsh_4);
            functions_.push_back(calculate_solid_mcsh_5);
            functions_.push_back(calculate_solid_mcsh_6);
            functions_.push_back(calculate_solid_mcsh_7);
            functions_.push_back(calculate_solid_mcsh_8);
            functions_.push_back(calculate_solid_mcsh_9);
            
            // Reserve space for values
            num_values_.reserve(10);
            
            // Add values one by one            
            num_values_.push_back(1);  // for order 0
            num_values_.push_back(3);  // for order 1
            num_values_.push_back(6);  // for order 2
            num_values_.push_back(10); // for order 3
            num_values_.push_back(15); // for order 4
            num_values_.push_back(21); // for order 5
            num_values_.push_back(28); // for order 6
            num_values_.push_back(36); // for order 7
            num_values_.push_back(45); // for order 8
            num_values_.push_back(55); // for order 9
        }
        
    private: 
        std::vector<solid_gmp_function_t> functions_;
        std::vector<int> num_values_;
    };
    
    // functions 
    double weighted_square_sum(const int mcsh_order, const vec<double>& v);
}}