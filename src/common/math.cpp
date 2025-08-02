#include "math.hpp"

namespace gmp { namespace math {

    // check if a value is close to zero
    template <typename T>
    GPU_HOST_DEVICE bool isZero(T value) {
        return std::abs(value) < std::numeric_limits<T>::epsilon();
    }

    // check if two values are equal
    template<typename T>
    GPU_HOST_DEVICE typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b) {
        return a == b; // exact comparison for non-floating-point types
    }

    // Floating-point specialization with relative+absolute tolerance
    template<typename T>
    GPU_HOST_DEVICE typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    isEqual(const T& a, const T& b,
            T absTol, T relTol) {
        return std::fabs(a - b) <= std::max(absTol, relTol * std::max(std::fabs(a), std::fabs(b)));
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

    // Function template instantiations
    template float weighted_square_sum(const int mcsh_order, const vector<float>& v);
    template double weighted_square_sum(const int mcsh_order, const vector<double>& v);

    template float round_to_0_1(const float x);
    template double round_to_0_1(const double x);

    // Explicit template instantiations for isZero
    template bool isZero<float>(float);
    template bool isZero<double>(double);

}} 