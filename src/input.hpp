#pragma once
#include <string>
#include <unordered_map>
#include "math.hpp"
#include "types.hpp"
#include "error.hpp"
#include "atom.hpp"

namespace gmp { namespace input {

    using namespace gmp::math;
    using namespace gmp::containers;
    using namespace gmp::atom;

    // enums 
    enum class cutoff_method_t { custom_cutoff, cutoff_sigma, cutoff_sigma_elemental, 
        cutoff_feature_elemental, cutoff_feature_gaussian};
    enum class scaling_mode_t { radial, both };

    // to_string 
    inline std::string gmp_to_string(cutoff_method_t cutoff_method) {
        switch (cutoff_method) {
            case cutoff_method_t::custom_cutoff: return "custom_cutoff";
            case cutoff_method_t::cutoff_sigma: return "cutoff_sigma";
            case cutoff_method_t::cutoff_sigma_elemental: return "cutoff_sigma_elemental";
            case cutoff_method_t::cutoff_feature_elemental: return "cutoff_feature_elemental";
            case cutoff_method_t::cutoff_feature_gaussian: return "cutoff_feature_gaussian";
        }
        return "";
    }
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
        std::string output_file_;
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
        void print_config() const;
    };

    // descriptor configuration
    struct feature_t {
        double sigma;
        int order;
        feature_t(int order, double sigma) : sigma(sigma), order(order) {}

        bool operator<(const feature_t& other) const {
            return (sigma < other.sigma) || (sigma == other.sigma && order < other.order);
        }
    };

    class descriptor_config_t {
    public: 
        descriptor_config_t() : feature_list_(), cutoff_method_(cutoff_method_t::cutoff_feature_gaussian), 
            scaling_mode_(scaling_mode_t::radial), cutoff_(0.0), overlap_threshold_(1e-11), square_(false) {}
        ~descriptor_config_t() = default;

    private:
        vec<feature_t> feature_list_;
        cutoff_method_t cutoff_method_;
        scaling_mode_t scaling_mode_;
        double cutoff_;
        double overlap_threshold_;
        bool square_;

    public:
        // accessor
        const vec<feature_t>& get_feature_list() const { return feature_list_; }
        cutoff_method_t get_cutoff_method() const { return cutoff_method_; }
        scaling_mode_t get_scaling_mode() const { return scaling_mode_; }
        double get_cutoff() const { return cutoff_; }
        double get_overlap_threshold() const { return overlap_threshold_; }
        bool get_square() const { return square_; }

        // setter
        void set_feature_list(const std::vector<int> orders, const std::vector<double> sigmas, const std::vector<std::tuple<int, double>> feature_list);
        void set_cutoff_method(const cutoff_method_t cutoff_method) { cutoff_method_ = cutoff_method; }
        void set_scaling_mode(const scaling_mode_t scaling_mode) { scaling_mode_ = scaling_mode; }
        void set_cutoff(const double cutoff) { cutoff_ = cutoff; }
        void set_overlap_threshold(const double overlap_threshold) { overlap_threshold_ = overlap_threshold; }
        void set_square(const bool square) { square_ = square; }

        // print config
        void print_config() const;
    };
    
    struct gaussian_t {
        double B;
        double beta;
        gaussian_t(double B, double beta) : B(B), beta(beta) {}
    };
    
    // psp config 
    class psp_config_t {
    public:
        psp_config_t() : gaussian_table_() {}
        psp_config_t(vec<vec<gaussian_t>>&& gaussian_table) : gaussian_table_(std::move(gaussian_table)) {}
        ~psp_config_t() = default;

        int get_num_gaussians_offset(vec<int>& offset) const;
        int size() const { return gaussian_table_.size(); }
        const vec<gaussian_t>& operator[](atom_type_id_t id) const { return gaussian_table_[id]; }

        // iterators
        auto begin() { return gaussian_table_.begin(); }
        auto end() { return gaussian_table_.end(); }
        auto begin() const { return gaussian_table_.begin(); }
        auto end() const { return gaussian_table_.end(); }
    
    private: 
        vec<vec<gaussian_t>> gaussian_table_;
        void print_config() const;
    };

    // input class
    class input_t {
    public:
        input_t() : files(make_gmp_unique<file_path_t>()), descriptor_config(make_gmp_unique<descriptor_config_t>()) {}
        ~input_t() = default;

    public: 
        // input paths
        gmp_unique_ptr<file_path_t> files;

        // descriptor_config
        gmp_unique_ptr<descriptor_config_t> descriptor_config;

    public: 
        // functions
        void parse_arguments(int argc, char* argv[]);

        void print_config() const;
    
    private: 
        // helper for argument
        void print_help() const;
    };
    
    // read psp file
    void read_psp_file(const std::string& psp_file, const atom_type_map_t& atom_type_map, vec<vec<gaussian_t>>& gaussian_table);

    // read atom file
    void read_atom_file(const std::string& atom_file, lattice_t& lattice, vec<atom_t>& atoms, atom_type_map_t& atom_type_map);

    // reference configuration
    struct reference_config_t {
        vec<array3d_flt64> ref_positions;
        double largest_cutoff = 0.0;
    };

}}