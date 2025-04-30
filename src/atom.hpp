#pragma once
#include <string>
#include "geometry.hpp"
#include "fwd.hpp"
#include "resources.hpp"

namespace gmp { namespace atom {
    
    using namespace gmp::geometry;
    using namespace gmp::resources;

    class atom_t {
    public:        
        // ctor
        atom_t(const double x, const double y, const double z, const double occupancy = 1.0, const std::string& type = "X")
            : position{x, y, z}, occupancy{occupancy}, type{type} {}
        atom_t(const point_flt64 position, const double occupancy = 1.0, const std::string& type = "X")
            : position{position}, occupancy{occupancy}, type{type} {}
        atom_t(const atom_t& other)
            : position{other.position}, occupancy{other.occupancy}, type{other.type} {}
        atom_t(atom_t&& other) noexcept
            : position{std::move(other.position)}, occupancy{std::move(other.occupancy)}, type{std::move(other.type)} {}
        
        // copy assignment operator
        atom_t& operator=(const atom_t& other) {
            position = other.position;
            occupancy = other.occupancy;
            type = other.type;
            return *this;
        }
        atom_t& operator=(atom_t&& other) noexcept {
            position = std::move(other.position);
            occupancy = std::move(other.occupancy);
            type = std::move(other.type);
            return *this;
        }

        // destructor
        ~atom_t() = default;

        // accessors
        const point_flt64& get_position() const { return position; }
        const double get_occupancy() const { return occupancy; }
        const std::string& get_type() const { return type; }
        const double x() const { return position.x; }
        const double y() const { return position.y; }
        const double z() const { return position.z; }

    private:
        point_flt64 position;
        double occupancy;
        std::string type;
    };
    
    class unit_cell_t {
    public:        
        // ctor 
        unit_cell_t()
            : atoms{gmp_resource::instance().get_host_memory().get_allocator<atom_t>()}, 
              lattice{gmp::resources::make_pool_unique<lattice_t>(gmp_resource::instance().get_host_memory().get_pool())}, 
              periodicity{true, true, true} {}

        unit_cell_t(vec<atom_t>&& atoms, gmp_unique_ptr<lattice_t>&& lattice, const array3d_bool& periodicity = array3d_bool{true, true, true})
            : atoms{std::move(atoms)}, 
              lattice{std::move(lattice)}, 
              periodicity{periodicity} {}

        // move constructor  
        unit_cell_t(unit_cell_t&& other) noexcept
            : atoms{std::move(other.atoms)},
              lattice{std::move(other.lattice)},
              periodicity{std::move(other.periodicity)} {}

        // move assignment
        unit_cell_t& operator=(unit_cell_t&& other) noexcept {
            if (this != &other) {
                atoms = std::move(other.atoms);
                lattice = std::move(other.lattice);
                periodicity = std::move(other.periodicity);
            }
            return *this;
        }
        
        // there is only 1 system, so no copy constructor or assignment
        unit_cell_t(const unit_cell_t&) = delete;
        unit_cell_t& operator=(const unit_cell_t&) = delete;

        // destructor
        ~unit_cell_t() = default;

        // accessors
        const vec<atom_t>& get_atoms() const { return atoms; }
        const gmp_unique_ptr<lattice_t>& get_lattice() const { return lattice; }
        const array3d_bool& get_periodicity() const { return periodicity; }
        atom_t& operator[](size_t i) { return atoms[i]; }
        const atom_t& operator[](size_t i) const { return atoms[i]; }

        // mutators
        void set_atoms(vec<atom_t>&& atoms) { this->atoms = std::move(atoms); }
        void set_lattice(gmp_unique_ptr<lattice_t>&& lattice) { this->lattice = std::move(lattice); }
        void set_periodicity(const array3d_bool& periodicity) { this->periodicity = periodicity; }

    private:
        vec<atom_t> atoms;
        gmp_unique_ptr<lattice_t> lattice;
        array3d_bool periodicity;
    };

}}