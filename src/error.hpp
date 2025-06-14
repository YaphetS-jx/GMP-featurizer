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
