#pragma once
#include <iostream>
#include "fwd.hpp"

#define GMP_CHECK(val)                                                      \
    do {                                                                    \
        if ((val) != error_t::success) {                                    \
            std::cerr << "GMP Function \"" << __func__                      \
                        << "\" failed at " << __FILE__ << ":" << __LINE__   \
                        << " with error: " << to_string(val) << std::endl;  \
            exit(static_cast<int>(val));                                    \
        }                                                                   \
    } while (0)

namespace gmp {

    inline const char* to_string(error_t err) {
        switch (err) {
            case error_t::success: return "Success";
            case error_t::matrix_singular: return "Matrix is singular";
            default: return "Unknown Error";
        }
    }
}