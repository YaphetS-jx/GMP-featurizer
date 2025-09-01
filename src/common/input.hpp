#pragma once
#include <string>
#include <unordered_map>
#include "math.hpp"
#include "containers.hpp"
#include "error.hpp"
#include "atom.hpp"
#include "gmp_float.hpp"

namespace gmp { namespace input {

    using namespace gmp::math;
    using namespace gmp::containers;
    using namespace gmp::atom;

    // enums 
    enum class scaling_mode_t { radial, both };

    // to_string 
    inline std::string gmp_to_string(scaling_mode_t scaling_mode) {
        switch (scaling_mode) {
            case scaling_mode_t::radial: return "radial";
            case scaling_mode_t::both: return "both";
        }
        return "";
    }

    // file path class
    class file_path_t {
    private:
        std::string atom_file_;
        std::string psp_file_;
        std::string output_file_{"./gmpFeatures.dat"};
    public:
        // accessor
        const std::string& get_atom_file() const { return atom_file_; }
        const std::string& get_psp_file() const { return psp_file_; }
        const std::string& get_output_file() const { return output_file_; }

        // setter
        void set_atom_file(const std::string& atom_file) { this->atom_file_ = atom_file; }
        void set_psp_file(const std::string& psp_file) { this->psp_file_ = psp_file; }
        void set_output_file(const std::string& output_file) { this->output_file_ = output_file; }

        // print config
        void dump() const;
    };

    // descriptor configuration
    template <typename T>
    struct feature_t {
        T sigma;
        int order;
        feature_t(int order, T sigma) : sigma(sigma), order(order) {}

        bool operator<(const feature_t& other) const {
            return (sigma < other.sigma) || (sigma == other.sigma && order < other.order);
        }
    };

    template <typename T>
    class descriptor_config_t {
    public: 
        descriptor_config_t() : feature_list_(), 
            scaling_mode_(scaling_mode_t::radial), 
            ref_grid_({0, 0, 0}),
            overlap_threshold_(1e-11), square_(false),
            num_bits_per_dim_(5), num_threads_(0), enable_gpu_(true) {}
        ~descriptor_config_t() = default;

    private:
        std::vector<feature_t<T>> feature_list_;
        scaling_mode_t scaling_mode_;
        array3d_int32 ref_grid_;
        T overlap_threshold_;
        bool square_;
        uint8_t num_bits_per_dim_;
        size_t num_threads_;
        bool enable_gpu_;

    public:
        // accessor
        const std::vector<feature_t<T>>& get_feature_list() const { return feature_list_; }
        scaling_mode_t get_scaling_mode() const { return scaling_mode_; }        
        T get_overlap_threshold() const { return overlap_threshold_; }
        bool get_square() const { return square_; }
        const array3d_int32& get_ref_grid() const { return ref_grid_; }
        uint8_t get_num_bits_per_dim() const { return num_bits_per_dim_; }
        size_t get_num_threads() const { return num_threads_; }
        bool get_enable_gpu() const { return enable_gpu_; }

        // setter
        void set_feature_list(const std::vector<int> orders, const std::vector<T> sigmas, const std::vector<std::tuple<int, T>> feature_list);
        void set_scaling_mode(const scaling_mode_t scaling_mode) { scaling_mode_ = scaling_mode; }
        void set_overlap_threshold(const T overlap_threshold) { overlap_threshold_ = overlap_threshold; }
        void set_square(const bool square) { square_ = square; }
        void set_ref_grid(const array3d_int32& ref_grid) { ref_grid_ = ref_grid; }
        void set_num_bits_per_dim(const uint8_t num_bits_per_dim) { num_bits_per_dim_ = num_bits_per_dim; }
        void set_num_threads(const size_t num_threads) { num_threads_ = num_threads; }
        void set_enable_gpu(const bool enable_gpu) { enable_gpu_ = enable_gpu; }

        // print config
        void dump() const;
    };

    // Type aliases using configured floating-point type
    using descriptor_config_flt = descriptor_config_t<gmp::gmp_float>;
    using feature_flt = feature_t<gmp::gmp_float>;

    // input class
    class input_t {
    public:
        input_t(const std::string& json_file);
        ~input_t() = default;

    public: 
        // input paths
        std::unique_ptr<file_path_t> files;

        // descriptor_config
        std::unique_ptr<descriptor_config_flt> descriptor_config;

    public: 
        // functions
        const descriptor_config_flt* get_descriptor_config() const { return descriptor_config.get(); }

        void parse_json(const std::string& json_file);

        void dump() const;
    
    private: 
        // helper for argument
        void print_help() const;
    };
}}