#pragma once
#include <cstdint>
#include <array>
#include <cmath>
#include "math.hpp"

namespace gmp { namespace geometry {

    using namespace gmp::math;

    // 3d point class
    template <typename T>
    struct point3d_t {
        T x, y, z;
    };

    // type aliases
    using point_int32 = point3d_t<int32_t>;
    using point_flt64 = point3d_t<double>;

    // Lattice system - represents a crystal lattice with 3 basis vectors
    class lattice_t {
    public:
        // constructor
        lattice_t() : lattice_vectors{array3d_flt64{}, array3d_flt64{}, array3d_flt64{}} {}
        
        // constructor with unit lattice vectors and cell length
        lattice_t(const array3d_flt64& v1, const array3d_flt64& v2, const array3d_flt64& v3, 
                double a = 1.0, double b = 1.0, double c = 1.0) 
            : lattice_vectors{v1*a, v2*b, v3*c} {}
        
        lattice_t(const matrix3d_flt64& lattice_vectors)
            : lattice_vectors{lattice_vectors} {}

        lattice_t(const lattice_t& other)
            : lattice_vectors{other.lattice_vectors} {}
        lattice_t(lattice_t&& other) noexcept
            : lattice_vectors{std::move(other.lattice_vectors)} {}
        
        // assignment operator
        lattice_t& operator=(const lattice_t& other) {
            lattice_vectors = other.lattice_vectors;
            return *this;
        }
        lattice_t& operator=(lattice_t&& other) noexcept {
            lattice_vectors = std::move(other.lattice_vectors);
            return *this;
        }
        
        // Access lattice vectors
        const array3d_flt64& operator[](size_t i) const { return lattice_vectors[i]; }
        array3d_flt64& operator[](size_t i) { return lattice_vectors[i]; }

        // get inverse lattice vectors
        matrix3d_flt64 get_inverse_lattice_vector() const {            
            return lattice_vectors.inverse();
        }

        // get volume of cell
        double get_volume() const {
            return std::abs(lattice_vectors.det());
        }

        // normalize lattice vectors to unit length
        matrix3d_flt64 normalize() const {
            matrix3d_flt64 result;
            for(int i = 0; i < 3; i++) {
                result[i] = lattice_vectors[i] / lattice_vectors[i].norm();
            }
            return result;
        }

        // get lattice metric
        sym_matrix3d_flt64 get_metric() const {
            return sym_matrix3d_flt64(lattice_vectors[0].dot(lattice_vectors[0]),
                                  lattice_vectors[1].dot(lattice_vectors[1]),
                                  lattice_vectors[2].dot(lattice_vectors[2]),
                                  lattice_vectors[0].dot(lattice_vectors[1]),
                                  lattice_vectors[0].dot(lattice_vectors[2]),
                                  lattice_vectors[1].dot(lattice_vectors[2]));
        }

    private:
        matrix3d_flt64 lattice_vectors;
    };

    // Convert cell parameters to lattice vectors
    lattice_t cell_info_to_lattice(const array3d_flt64& cell_lengths, const array3d_flt64& cell_angles);

}}