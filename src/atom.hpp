#pragma once
#include <string>
#include <unordered_map>
#include "geometry.hpp"
#include "types.hpp"
#include "resources.hpp"

namespace gmp { namespace atom {
    
    using namespace gmp::geometry;    
    using namespace gmp::containers;

    class atom_t {
    public:        
        // ctor
        atom_t(const double x, const double y, const double z, const double occupancy = 1.0, const atom_type_id_t type_id = std::numeric_limits<atom_type_id_t>::max())
            : position_{x, y, z}, occupancy_(occupancy), type_id_(type_id) {}
        atom_t(const point_flt64 position, const double occupancy = 1.0, const atom_type_id_t type_id = std::numeric_limits<atom_type_id_t>::max())
            : position_(position), occupancy_(occupancy), type_id_(type_id) {}
        // destructor
        ~atom_t() = default;

        // accessors
        const point_flt64& pos() const { return position_; }
        double occ() const { return occupancy_; }
        atom_type_id_t id() const { return type_id_; }
        double x() const { return position_.x; }
        double y() const { return position_.y; }
        double z() const { return position_.z; }

        // mutators
        void set_pos(const point_flt64& position) { position_ = position; }
        void set_occ(const double occupancy) { occupancy_ = occupancy; }
        void set_id(const atom_type_id_t type_id) { type_id_ = type_id; }

    private:
        point_flt64 position_;
        double occupancy_;
        atom_type_id_t type_id_;
    };
    
    class unit_cell_t {
    public:
        unit_cell_t() = default;
        unit_cell_t(std::string atom_file);
        ~unit_cell_t() = default;

        // accessors
        const vec<atom_t>& get_atoms() const { return atoms_; }
        const lattice_t* get_lattice() const { return lattice_.get(); }
        const array3d_bool& get_periodicity() const { return periodicity_; }
        const atom_t& operator[](size_t i) const { return atoms_[i]; }
        const atom_type_map_t& get_atom_type_map() const { return atom_type_map_; }

        // mutators
        void set_lattice(std::unique_ptr<lattice_t>&& lattice) { lattice_ = std::move(lattice); }
        void set_atoms(vec<atom_t>&& atoms) { atoms_ = std::move(atoms); }
        void set_atom_type_map(atom_type_map_t&& atom_type_map) { atom_type_map_ = std::move(atom_type_map); }
        void set_periodicity(const array3d_bool& periodicity) { periodicity_ = periodicity; }

        void dump() const;

    private:
        std::unique_ptr<lattice_t> lattice_;
        vec<atom_t> atoms_;
        atom_type_map_t atom_type_map_;
        array3d_bool periodicity_;
    };

    struct gaussian_t {
        double B;
        double beta;
        gaussian_t(double B, double beta) : B(B), beta(beta) {}
    };
    
    // psp config 
    class psp_config_t {
    public:
        psp_config_t(std::string psp_file, const unit_cell_t* unit_cell);
        ~psp_config_t() = default;

        const vec<int>& get_offset() const { return offset_; }        
        const gaussian_t& operator[](const int idx) const { return gaussian_table_[idx]; }
        void dump() const;

    private:         
        vec<gaussian_t> gaussian_table_;
        vec<int> offset_;
    };

    // read psp file
    void read_psp_file(const std::string& psp_file, const atom_type_map_t& atom_type_map, vec<gaussian_t>& gaussian_table, vec<int>& offset);

    // read atom file
    void read_atom_file(const std::string& atom_file, std::unique_ptr<lattice_t>& lattice, vec<atom_t>& atoms, atom_type_map_t& atom_type_map);

    // set reference positions
    vec<point_flt64> set_ref_positions(const array3d_int32& ref_grid, const vec<atom_t>& atoms);
}}