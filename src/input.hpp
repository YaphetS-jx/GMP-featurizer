#pragma once
#include <string>
#include "math.hpp"
#include "types.hpp"

namespace gmp { namespace input {

    using namespace gmp::math;
    using namespace gmp::containers;

    // enums 
    enum class cutoff_method_t { custom_cutoff, cutoff_sigma, cutoff_sigma_widest_gaussian, 
        cutoff_order_sigma_widest_gaussian, cutoff_order_sigma_each_gaussian};    
    enum class scaling_mode_t { radial, both };

    // basic structures
    struct feature_t {
        double sigma;
        int order;
    };
    struct KernelParams {
        double C1, C2;
        double lambda, gamma;
    };
    struct file_path_t {
        std::string atom_file;
        std::string psp_file;
        std::string output_file;
    };

    // input class
    class input_t {
    public:
        input_t() = default;
        ~input_t() = default;

        // input paths
        gmp_unique_ptr<file_path_t> files;

        // 
    public: 
        // functions
        void parse_arguments(int argc, char* argv[]);
    
    private: 
        // helper for argument
        void print_help() const;
    };
    
    // descriptor configuration
    struct descriptor_config_t {
        vec<feature_t> feature_list;
        cutoff_method_t cutoff_method = cutoff_method_t::cutoff_order_sigma_each_gaussian;
        scaling_mode_t scaling_mode = scaling_mode_t::radial;
        double cutoff = 0.0;
        double overlap_threshold = 1e-11;
        bool square = false;
    };

    // reference configuration
    struct reference_config_t {
        vec<array3d_flt64> ref_positions;
        double largest_cutoff = 0.0;
    };

}}