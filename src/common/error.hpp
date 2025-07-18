#pragma once
#include <iostream>

namespace gmp {

    enum class error_t {
        success,
        // memory errors
        memory_bad_alloc,
        // input errors 
        invalid_argument,
        invalid_json_file,
        invalid_cutoff_method,
        invalid_scaling_mode,
        invalid_feature_list,
        invalid_order_sigma,
        unstreamable_string,
        incomplete_data_set,
        invalid_psp_file,
        invalid_atom_file,
        missing_atom_psp, 
        // math errors
        matrix_singular,
        invalid_mcsh_order,
        // output errors
        output_file_error,
        // tree errors
        tree_invalid_morton_code,
    };

    // Global error variable declaration
    extern error_t gmp_error;

    // Function declarations
    void update_error(error_t err);
    const char* gmp_to_string(error_t err);
    error_t get_last_error();

    // Global error variable definition with initial value
    inline error_t gmp_error = error_t::success;

    // Update the error code
    inline void update_error(error_t err) {
        gmp_error = err;
    }

    // Convert error code to string
    inline const char* gmp_to_string(error_t err) {
        switch (err) {
            case error_t::success: return "Success";
            // memory errors
            case error_t::memory_bad_alloc: return "Memory Allocation Failed";
            // input errors
            case error_t::invalid_json_file: return "Invalid JSON File";
            case error_t::invalid_argument: return "Invalid Argument";
            case error_t::invalid_cutoff_method: return "Invalid Cutoff Method";
            case error_t::invalid_scaling_mode: return "Invalid Scaling Mode";
            case error_t::invalid_feature_list: return "Invalid Feature List";
            case error_t::invalid_order_sigma: return "Invalid Order or Sigma";
            case error_t::unstreamable_string: return "Unstreamable String";
            case error_t::incomplete_data_set: return "Incomplete Data Set";
            case error_t::invalid_psp_file: return "Invalid PSP File";
            case error_t::invalid_atom_file: return "Invalid Atom File";
            case error_t::missing_atom_psp: return "Missing Atom PSP";
            // math errors
            case error_t::matrix_singular: return "Matrix is singular";
            case error_t::invalid_mcsh_order: return "Invalid MCSH Order";
            // output errors
            case error_t::output_file_error: return "Output File Error";
            default: return "Unknown Error";
        }
    }

    // Get last error
    inline error_t get_last_error() {
        return gmp_error;
    }

} // namespace gmp

#define GMP_CHECK(val)                                                          \
    do {                                                                        \
        if ((val) != gmp::error_t::success) {                                   \
            std::cerr << "GMP Function \"" << __func__                          \
                        << "\" failed at " << __FILE__ << ":" << __LINE__       \
                        << " with error: " << gmp::gmp_to_string(val) << std::endl; \
            exit(static_cast<int>(val));                                        \
        }                                                                       \
    } while (0)

#define GMP_EXIT(val)                                                           \
    do {                                                                        \
        std::cerr << "GMP Function \"" << __func__                              \
                    << "\" failed at " << __FILE__ << ":" << __LINE__           \
                    << " with error: " << gmp::gmp_to_string(val) << std::endl;     \
        exit(static_cast<int>(val));                                            \
    } while (0)
