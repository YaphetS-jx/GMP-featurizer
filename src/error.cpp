#include "error.hpp"

namespace gmp {

    // Global error variable definition with initial value
    error_t gmp_error = error_t::success;

    // Update the error code
    void update_error(error_t err) {
        gmp_error = err;
    }

    // Convert error code to string
    const char* to_string(error_t err) {
        switch (err) {
            case error_t::success: return "Success";
            case error_t::memory_bad_alloc: return "Memory Allocation Failed";
            case error_t::invalid_argument: return "Invalid Argument";
            case error_t::unstreamable_string: return "Unstreamable String";
            case error_t::incomplete_data_set: return "Incomplete Data Set";
            case error_t::matrix_singular: return "Matrix is singular";
            default: return "Unknown Error";
        }
    }

    // Get last error
    error_t get_last_error() {
        return gmp_error;
    }

} // namespace gmp 