#pragma once
#include <string>
#include "geometry.hpp"
#include "types.hpp"
#include "resources.hpp"

namespace gmp { namespace atom {
    
    using namespace gmp::geometry;    
    using namespace gmp::containers;

    class atom_t {
    public:        
        // ctor
        atom_t(const double x, const double y, const double z, const double occupancy = 1.0, const std::string& type = "X")
            : position_{x, y, z}, occupancy_{occupancy}, type_{type} {}
        atom_t(const point_flt64 position, const double occupancy = 1.0, const std::string& type = "X")
            : position_{position}, occupancy_{occupancy}, type_{type} {}
        atom_t(const atom_t& other)
            : position_{other.position_}, occupancy_{other.occupancy_}, type_{other.type_} {}
        atom_t(atom_t&& other) noexcept
            : position_{std::move(other.position_)}, occupancy_{std::move(other.occupancy_)}, type_{std::move(other.type_)} {}
        
        // copy assignment operator
        atom_t& operator=(const atom_t& other) {
            position_ = other.position_;
            occupancy_ = other.occupancy_;
            type_ = other.type_;
            return *this;
        }
        atom_t& operator=(atom_t&& other) noexcept {
            position_ = std::move(other.position_);
            occupancy_ = std::move(other.occupancy_);
            type_ = std::move(other.type_);
            return *this;
        }

        // destructor
        ~atom_t() = default;

        // accessors
        const point_flt64& get_position() const { return position_; }
        double get_occupancy() const { return occupancy_; }
        const std::string& get_type() const { return type_; }
        double x() const { return position_.x; }
        double y() const { return position_.y; }
        double z() const { return position_.z; }

    private:
        point_flt64 position_;
        double occupancy_;
        std::string type_;
    };
    
    class unit_cell_t {
    public:        
        // ctor 
        unit_cell_t()
            : atoms_{}, lattice_{}, periodicity_{true, true, true} {}

        unit_cell_t(vec<atom_t>&& atoms, gmp_unique_ptr<lattice_t>&& lattice, const array3d_bool& periodicity = array3d_bool{true, true, true})
            : atoms_{std::move(atoms)}, lattice_{std::move(lattice)}, periodicity_{periodicity} {}

        // move constructor  
        unit_cell_t(unit_cell_t&& other) noexcept
            : atoms_{std::move(other.atoms_)}, lattice_{std::move(other.lattice_)}, periodicity_{std::move(other.periodicity_)} {}

        // move assignment
        unit_cell_t& operator=(unit_cell_t&& other) noexcept {
            if (this != &other) {
                atoms_ = std::move(other.atoms_);
                lattice_ = std::move(other.lattice_);
                periodicity_ = std::move(other.periodicity_);
            }
            return *this;
        }
        
        // there is only 1 system, so no copy constructor or assignment
        unit_cell_t(const unit_cell_t&) = delete;
        unit_cell_t& operator=(const unit_cell_t&) = delete;

        // destructor
        ~unit_cell_t() = default;

        // accessors
        const vec<atom_t>& get_atoms() const { return atoms_; }
        const gmp_unique_ptr<lattice_t>& get_lattice() const { return lattice_; }
        const array3d_bool& get_periodicity() const { return periodicity_; }
        atom_t& operator[](size_t i) { return atoms_[i]; }
        const atom_t& operator[](size_t i) const { return atoms_[i]; }

        // mutators
        void set_atoms(vec<atom_t>&& atoms) { this->atoms_ = std::move(atoms); }
        void set_lattice(gmp_unique_ptr<lattice_t>&& lattice) { this->lattice_ = std::move(lattice); }
        void set_periodicity(const array3d_bool& periodicity) { this->periodicity_ = periodicity; }

    private:
        vec<atom_t> atoms_;
        gmp_unique_ptr<lattice_t> lattice_;
        array3d_bool periodicity_;
    };

}}