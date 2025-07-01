#pragma once
#include "mcsh.hpp"
#include "mcsh_kernel.hxx"

namespace gmp { namespace mcsh {
    // register functions
    template <typename T>
    mcsh_registry_t<T>::mcsh_registry_t() : 
        functions_{ solid_mcsh_0<T>,
                    solid_mcsh_1<T>,
                    solid_mcsh_2<T>,
                    solid_mcsh_3<T>,
                    solid_mcsh_4<T>,
                    solid_mcsh_5<T>,
                    solid_mcsh_6<T>,
                    solid_mcsh_7<T>,
                    solid_mcsh_8<T>,
                    solid_mcsh_9<T>
        },
        num_values_{1, 3, 6, 10, 15, 21, 28, 36, 45} 
    {}

    template <typename T>
    mcsh_registry_t<T>& mcsh_registry_t<T>::get_instance() {
        static mcsh_registry_t<T> instance;
        return instance;
    }

    template <typename T>
    const typename mcsh_registry_t<T>::solid_gmp_t mcsh_registry_t<T>::get_function(int order) const {
        return order < 0 ? functions_[0] : functions_[order];
    }

    template <typename T>
    int mcsh_registry_t<T>::get_num_values(int order) const {
        return order < 0 ? num_values_[0] : num_values_[order];
    }

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
}}