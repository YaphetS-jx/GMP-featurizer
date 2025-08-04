#pragma once
#include <cstdint>
#include <array>
#include <cmath>
#include <memory>
#include <iostream>
#include "math.hpp"
#include "gmp_float.hpp"

namespace gmp { namespace geometry {

    // Type aliases for external structures
    using gmp::math::array3d_t;
    using gmp::math::matrix3d_t;
    using gmp::math::sym_matrix3d_t;
    using gmp::math::array3d_flt;
    using gmp::math::matrix3d_flt;
    using gmp::math::sym_matrix3d_flt;

    // 3d point class
    template <typename T>
    struct point3d_t {
        T x, y, z;
    };

    // Stream insertion operator for point3d_t
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const point3d_t<T>& p);

    // type aliases using configured floating-point type
    using point_int32 = point3d_t<int32_t>;
    using point_flt = point3d_t<gmp::gmp_float>;

    // Lattice system - represents a crystal lattice with 3 basis vectors
    template <typename T>
    class lattice_t {
    public:
        using array3d_type = array3d_t<T>;
        using matrix3d_type = matrix3d_t<T>;
        using sym_matrix3d_type = sym_matrix3d_t<T>;
        using point_type = point3d_t<T>;

        lattice_t(matrix3d_type lattice_vectors);
        
        // Access lattice vectors
        const array3d_type& operator[](size_t i) const;
        array3d_type& operator[](size_t i);

        // get lattice metric
        void update_metric();

        // get volume of cell
        T get_volume() const;

        // normalize lattice vectors to unit length
        matrix3d_type normalize() const;

        // get inter-planar spacing
        array3d_type get_interplanar_spacing() const;

        array3d_type get_cell_lengths() const;

        T calculate_distance_squared(const point_type& p1, const point_type& p2, const array3d_type& cell_shift, array3d_type& difference) const;

        T calculate_distance_squared(const array3d_type& difference) const;

        T calculate_distance(const point_type& p1, const point_type& p2, const array3d_type& cell_shift, array3d_type& difference) const;

        array3d_type fractional_to_cartesian(const array3d_type& fractional) const;

        void dump() const;

    private:
        matrix3d_type lattice_vectors_;
        // matrix3d_type inverse_lattice_vectors_;
        sym_matrix3d_type metric_;
    };

    // Convert cell parameters to lattice vectors
    template <typename T>
    std::unique_ptr<lattice_t<T>> cell_info_to_lattice(const array3d_t<T>& cell_lengths, const array3d_t<T>& cell_angles);

    // Type aliases using configured floating-point type
    using lattice_flt = lattice_t<gmp::gmp_float>;

}}