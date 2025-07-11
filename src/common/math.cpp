#include "math.hpp"

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
            T absTol, T relTol) {
        return std::fabs(a - b) <= std::max(absTol, relTol * std::max(std::fabs(a), std::fabs(b)));
    }

    // array_2d_t implementations
    template <typename T>
    array_2d_t<T>::array_2d_t() : data_{0, 0} {}

    template <typename T>
    array_2d_t<T>::array_2d_t(T x, T y) : data_{x, y} {}

    template <typename T>
    array_2d_t<T>::array_2d_t(const array_2d_t<T>& other) : data_{other.data_[0], other.data_[1]} {}

    template <typename T>
    array_2d_t<T>::array_2d_t(array_2d_t<T>&& other) noexcept : data_{other.data_[0], other.data_[1]} {}

    template <typename T>
    array_2d_t<T>& array_2d_t<T>::operator=(const array_2d_t<T>& other) {
        data_[0] = other.data_[0];
        data_[1] = other.data_[1];
        return *this;
    }

    template <typename T>
    array_2d_t<T>& array_2d_t<T>::operator=(array_2d_t<T>&& other) noexcept {
        if (this != &other) {
            data_[0] = other.data_[0];
            data_[1] = other.data_[1];
        }
        return *this;
    }

    template <typename T>
    T& array_2d_t<T>::operator[](size_t i) { 
        assert(i < 2); 
        return data_[i]; 
    }

    template <typename T>
    const T& array_2d_t<T>::operator[](size_t i) const { 
        assert(i < 2); 
        return data_[i]; 
    }

    // array3d_t implementations
    template <typename T>
    array3d_t<T>::array3d_t() : data_{0, 0, 0} {}

    template <typename T>
    array3d_t<T>::array3d_t(T x, T y, T z) : data_{x, y, z} {}

    template <typename T>
    array3d_t<T>::array3d_t(const array3d_t<T>& other) : data_{other.data_[0], other.data_[1], other.data_[2]} {}

    template <typename T>
    array3d_t<T>::array3d_t(array3d_t<T>&& other) noexcept : data_{other.data_[0], other.data_[1], other.data_[2]} {}

    template <typename T>
    array3d_t<T>& array3d_t<T>::operator=(const array3d_t<T>& other) {
        data_[0] = other.data_[0];
        data_[1] = other.data_[1];
        data_[2] = other.data_[2];
        return *this;
    }

    template <typename T>
    array3d_t<T>& array3d_t<T>::operator=(array3d_t<T>&& other) noexcept {
        if (this != &other) {
            data_[0] = other.data_[0];
            data_[1] = other.data_[1];
            data_[2] = other.data_[2];
        }
        return *this;
    }

    template <typename T>
    T& array3d_t<T>::operator[](size_t i) { 
        return data_[i]; 
    }

    template <typename T>
    const T& array3d_t<T>::operator[](size_t i) const { 
        return data_[i]; 
    }

    template <typename T>
    array3d_t<T> array3d_t<T>::operator+(const array3d_t<T>& other) const {
        return {data_[0] + other.data_[0], data_[1] + other.data_[1], data_[2] + other.data_[2]};
    }

    template <typename T>
    array3d_t<T>& array3d_t<T>::operator+=(const array3d_t<T>& other) {
        data_[0] += other.data_[0];
        data_[1] += other.data_[1];
        data_[2] += other.data_[2];
        return *this;
    }

    template <typename T>
    array3d_t<T> array3d_t<T>::operator-(const array3d_t<T>& other) const {
        return {data_[0] - other.data_[0], data_[1] - other.data_[1], data_[2] - other.data_[2]};
    }

    template <typename T>
    array3d_t<T>& array3d_t<T>::operator-=(const array3d_t<T>& other) {
        data_[0] -= other.data_[0];
        data_[1] -= other.data_[1];
        data_[2] -= other.data_[2];
        return *this;
    }

    template <typename T>
    array3d_t<T> array3d_t<T>::operator*(const T scalar) const {
        return {data_[0] * scalar, data_[1] * scalar, data_[2] * scalar};
    }
    
    template <typename T>
    array3d_t<T> array3d_t<T>::operator*(const array3d_t<T>& arr) const {
        return {data_[0] * arr[0], data_[1] * arr[1], data_[2] * arr[2]};
    }

    template <typename T>
    array3d_t<T> array3d_t<T>::operator/(const T scalar) const {
        return {data_[0] / scalar, data_[1] / scalar, data_[2] / scalar};
    }

    template <typename T>
    bool array3d_t<T>::operator==(const array3d_t<T>& other) const {
        return isEqual(data_[0], other.data_[0]) && isEqual(data_[1], other.data_[1]) && isEqual(data_[2], other.data_[2]);
    }

    template <typename T>
    bool array3d_t<T>::operator!=(const array3d_t<T>& other) const {
        return !(*this == other);
    }

    template <typename T>
    T array3d_t<T>::dot(const array3d_t<T>& other) const {
        return data_[0] * other.data_[0] + data_[1] * other.data_[1] + data_[2] * other.data_[2];
    }

    template <typename T>
    array3d_t<T> array3d_t<T>::cross(const array3d_t<T>& other) const {
        return {
            data_[1] * other.data_[2] - data_[2] * other.data_[1],
            data_[2] * other.data_[0] - data_[0] * other.data_[2],
            data_[0] * other.data_[1] - data_[1] * other.data_[0]
        };
    }

    template <typename T>
    T array3d_t<T>::norm() const {
        return std::sqrt(data_[0] * data_[0] + data_[1] * data_[1] + data_[2] * data_[2]);
    }

    template <typename T>
    T array3d_t<T>::prod() const {
        return data_[0] * data_[1] * data_[2];
    }

    // Standalone array3d_t operators
    template <typename T>
    array3d_t<T> operator*(const T scalar, const array3d_t<T>& arr) {
        return arr * scalar;
    }

    template <typename T>
    array3d_t<T> operator/(const T scalar, const array3d_t<T>& arr) {
        return {scalar / arr[0], scalar / arr[1], scalar / arr[2]};
    }

    // matrix3d_t implementations
    template <typename T>
    matrix3d_t<T>::matrix3d_t() : data_{array3d_t<T>{}, array3d_t<T>{}, array3d_t<T>{}} {}
    
    template <typename T>
    matrix3d_t<T>::matrix3d_t(const array3d_t<T>& row0, const array3d_t<T>& row1, const array3d_t<T>& row2) 
        : data_{row0, row1, row2} {}
            
    template <typename T>
    matrix3d_t<T>::matrix3d_t(const matrix3d_t<T>& other)
        : data_{other.data_[0], other.data_[1], other.data_[2]} {}

    template <typename T>
    matrix3d_t<T>::matrix3d_t(matrix3d_t<T>&& other) noexcept 
        : data_{other.data_[0], other.data_[1], other.data_[2]} {}

    template <typename T>
    matrix3d_t<T>& matrix3d_t<T>::operator=(const matrix3d_t<T>& other) {
        data_[0] = other.data_[0];
        data_[1] = other.data_[1];
        data_[2] = other.data_[2];
        return *this;
    }

    template <typename T>
    matrix3d_t<T>& matrix3d_t<T>::operator=(matrix3d_t<T>&& other) noexcept {
        if (this != &other) {
            data_[0] = other.data_[0];
            data_[1] = other.data_[1];
            data_[2] = other.data_[2];
        }
        return *this;
    }

    template <typename T>
    array3d_t<T>& matrix3d_t<T>::operator[](size_t i) { 
        return data_[i]; 
    }

    template <typename T>
    const array3d_t<T>& matrix3d_t<T>::operator[](size_t i) const { 
        return data_[i]; 
    }

    template <typename T>
    bool matrix3d_t<T>::operator==(const matrix3d_t<T>& other) const {
        return data_[0] == other.data_[0] && data_[1] == other.data_[1] && data_[2] == other.data_[2];
    }

    template <typename T>
    bool matrix3d_t<T>::operator!=(const matrix3d_t<T>& other) const {
        return !(*this == other);
    }

    template <typename T>
    matrix3d_t<T> matrix3d_t<T>::operator*(const matrix3d_t<T>& other) const {
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
    
    template <typename T>
    array3d_t<T> matrix3d_t<T>::operator*(const array3d_t<T>& other) const {
        array3d_t<T> result;
        for (int i = 0; i < 3; i++) {
            result[i] = data_[i].dot(other);
        }
        return result;
    }

    template <typename T>
    array3d_t<T> matrix3d_t<T>::transpose_mult(const array3d_t<T>& other) const {
        array3d_t<T> result;
        for (int i = 0; i < 3; i++) {
            result[i] = data_[0][i] * other[0] + data_[1][i] * other[1] + data_[2][i] * other[2];
        }
        return result;
    }

    template <typename T>
    matrix3d_t<T> matrix3d_t<T>::transpose() const {
        matrix3d_t<T> result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result[j][i] = data_[i][j];
            }
        }
        return result;
    }
    
    template <typename T>
    T matrix3d_t<T>::det() const {
        return data_[0][0]*(data_[1][1]*data_[2][2] - data_[1][2]*data_[2][1])
               - data_[0][1]*(data_[1][0]*data_[2][2] - data_[1][2]*data_[2][0])
               + data_[0][2]*(data_[1][0]*data_[2][1] - data_[1][1]*data_[2][0]);
    }

    template <typename T>
    matrix3d_t<T> matrix3d_t<T>::inverse() const {
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

    // sym_matrix3d_t implementations
    template <typename T>
    sym_matrix3d_t<T>::sym_matrix3d_t() : diag_{array3d_t<T>{}}, off_diag_{array3d_t<T>{}} {}

    template <typename T>
    sym_matrix3d_t<T>::sym_matrix3d_t(T a, T b, T c, T d, T e, T f) : diag_{array3d_t<T>{a, b, c}}, off_diag_{array3d_t<T>{d, e, f}} {}        

    template <typename T>
    sym_matrix3d_t<T>::sym_matrix3d_t(const array3d_t<T>& diag, const array3d_t<T>& off_diag) : diag_{diag}, off_diag_{off_diag} {}        

    template <typename T>
    sym_matrix3d_t<T>::sym_matrix3d_t(const sym_matrix3d_t<T>& other) : diag_{other.diag_}, off_diag_{other.off_diag_} {}

    template <typename T>
    sym_matrix3d_t<T>::sym_matrix3d_t(sym_matrix3d_t<T>&& other) noexcept : diag_{std::move(other.diag_)}, off_diag_{std::move(other.off_diag_)} {}
    
    template <typename T>
    sym_matrix3d_t<T>& sym_matrix3d_t<T>::operator=(const sym_matrix3d_t<T>& other) {
        diag_ = other.diag_;
        off_diag_ = other.off_diag_;
        return *this;
    }

    template <typename T>
    T& sym_matrix3d_t<T>::operator[](size_t i) { 
        if(i < 3) return diag_[i];
        return off_diag_[i-3];
    }

    template <typename T>
    const T& sym_matrix3d_t<T>::operator[](size_t i) const { 
        if(i < 3) return diag_[i];
        return off_diag_[i-3];
    }

    template <typename T>
    bool sym_matrix3d_t<T>::operator==(const sym_matrix3d_t<T>& other) const {
        return diag_ == other.diag_ && off_diag_ == other.off_diag_;
    }

    template <typename T>
    bool sym_matrix3d_t<T>::operator!=(const sym_matrix3d_t<T>& other) const {
        return !(*this == other);
    }

    template <typename T>
    array3d_t<T> sym_matrix3d_t<T>::operator*(const array3d_t<T>& vec) const {            
        array3d_t<T> result = { 
            diag_[0]*vec[0] + off_diag_[0]*vec[1] + off_diag_[1]*vec[2],
            off_diag_[0]*vec[0] + diag_[1]*vec[1] + off_diag_[2]*vec[2],
            off_diag_[1]*vec[0] + off_diag_[2]*vec[1] + diag_[2]*vec[2]
        };
        return result;
    }

    // other math functions 
    template <typename T>
    T round_to_0_1(const T x) {
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
    T weighted_square_sum(const int mcsh_order, const vector<T>& v) {
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
            update_error(error_t::invalid_mcsh_order);
            return 0.;
        }
    }

    // Explicit template instantiations
    template class array_2d_t<float>;
    template class array_2d_t<double>;

    template class array3d_t<float>;
    template class array3d_t<double>;
    template class array3d_t<uint32_t>;
    template class array3d_t<int32_t>;

    template class matrix3d_t<float>;
    template class matrix3d_t<double>;

    template class sym_matrix3d_t<float>;
    template class sym_matrix3d_t<double>;

    // Standalone operator instantiations
    template array3d_t<float> operator*(const float scalar, const array3d_t<float>& arr);
    template array3d_t<double> operator*(const double scalar, const array3d_t<double>& arr);

    template array3d_t<float> operator/(const float scalar, const array3d_t<float>& arr);
    template array3d_t<double> operator/(const double scalar, const array3d_t<double>& arr);

    // Function template instantiations
    template float weighted_square_sum(const int mcsh_order, const vector<float>& v);
    template double weighted_square_sum(const int mcsh_order, const vector<double>& v);

    template float round_to_0_1(const float x);
    template double round_to_0_1(const double x);

    // Explicit template instantiations for isZero
    template bool isZero<float>(float);
    template bool isZero<double>(double);

}} 