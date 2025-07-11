#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <array>
#include "geometry.hpp"
#include "types.hpp"
#include "resources.hpp"
#include "math.hpp"
#include "gmp_float.hpp"

namespace gmp { namespace atom {
    
    using gmp::geometry::lattice_flt;
    using gmp::geometry::point_flt;
    using gmp::geometry::point3d_t;
    using gmp::math::array3d_bool;
    using gmp::math::array3d_int32;
    using gmp::containers::vector;
    using gmp::containers::atom_type_map_t;
    using gmp::containers::atom_type_id_t;

    template <typename T>
    class atom_t {
    public:        
        using point_type = gmp::geometry::point3d_t<T>;
        
        // ctor
        atom_t(const T x, const T y, const T z, const T occupancy = static_cast<T>(1.0), const atom_type_id_t type_id = std::numeric_limits<atom_type_id_t>::max());
        atom_t(const point_type position, const T occupancy = static_cast<T>(1.0), const atom_type_id_t type_id = std::numeric_limits<atom_type_id_t>::max());
        // destructor
        ~atom_t() = default;

        // accessors
        const point_type& pos() const;
        T occ() const;
        atom_type_id_t id() const;
        T x() const;
        T y() const;
        T z() const;

        // mutators
        void set_pos(const point_type& position);
        void set_occ(const T occupancy);
        void set_id(const atom_type_id_t type_id);

    private:
        point_type position_;
        T occupancy_;
        atom_type_id_t type_id_;
    };
    
    template <typename T>
    class unit_cell_t {
    public:
        using lattice_type = gmp::geometry::lattice_t<T>;
        using point_type = gmp::geometry::point3d_t<T>;
        using atom_type = atom_t<T>;
        
        unit_cell_t();
        unit_cell_t(std::string atom_file);
        ~unit_cell_t() = default;

        // accessors
        const vector<atom_type>& get_atoms() const;
        const lattice_type* get_lattice() const;
        const array3d_bool& get_periodicity() const;
        const atom_type& operator[](size_t i) const;
        const atom_type_map_t& get_atom_type_map() const;

        // mutators
        void set_lattice(std::unique_ptr<lattice_type>&& lattice);
        void set_atoms(vector<atom_type>&& atoms);
        void set_atom_type_map(atom_type_map_t&& atom_type_map);
        void set_periodicity(const array3d_bool& periodicity);

        void dump() const;

    private:
        std::unique_ptr<lattice_type> lattice_;
        vector<atom_type> atoms_;
        atom_type_map_t atom_type_map_;
        array3d_bool periodicity_;
    };

    template <typename T>
    struct gaussian_t {
        T B;
        T beta;
        gaussian_t(T B, T beta);
    };
    
    // psp config 
    template <typename T>
    class psp_config_t {
    public:
        using gaussian_type = gaussian_t<T>;
        
        psp_config_t(std::string psp_file, const unit_cell_t<T>* unit_cell);
        ~psp_config_t() = default;

        const vector<int>& get_offset() const;
        const gaussian_type& operator[](const int idx) const;
        void dump() const;

    private:         
        vector<gaussian_type> gaussian_table_;
        vector<int> offset_;
    };

    // read psp file
    template <typename T>
    void read_psp_file(const std::string& psp_file, const atom_type_map_t& atom_type_map, vector<gaussian_t<T>>& gaussian_table, vector<int>& offset);

    // read atom file
    template <typename T>
    void read_atom_file(const std::string& atom_file, std::unique_ptr<gmp::geometry::lattice_t<T>>& lattice, vector<atom_t<T>>& atoms, atom_type_map_t& atom_type_map);

    // set reference positions
    template <typename T>
    vector<gmp::geometry::point3d_t<T>> set_ref_positions(const array3d_int32& ref_grid, const vector<atom_t<T>>& atoms);

    // Type aliases using configured floating-point type
    using atom_flt = atom_t<gmp::gmp_float>;
    using unit_cell_flt = unit_cell_t<gmp::gmp_float>;
    using gaussian_flt = gaussian_t<gmp::gmp_float>;
    using psp_config_flt = psp_config_t<gmp::gmp_float>;
}}