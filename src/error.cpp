#include "error.hpp"

namespace gmp {

    // Global error variable definition with initial value
    error_t gmp_error = error_t::success;

    // Update the error code
    void update_error(error_t err) {
        gmp_error = err;
    }

    // Convert error code to string
    const char* gmp_to_string(error_t err) {
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
    error_t get_last_error() {
        return gmp_error;
    }

} // namespace gmp 