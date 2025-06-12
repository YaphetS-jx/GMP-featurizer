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

    // Stream insertion operator for point3d_t
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const point3d_t<T>& p) {
        os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
        return os;
    }

    // type aliases
    using point_int32 = point3d_t<int32_t>;
    using point_flt64 = point3d_t<double>;

    // Lattice system - represents a crystal lattice with 3 basis vectors
    class lattice_t {
    public:
        lattice_t(matrix3d_flt64 lattice_vectors) : lattice_vectors_(lattice_vectors), metric_{} 
        {
            update_metric();
        }
        
        // copy constructor
        lattice_t(const lattice_t& other) : lattice_vectors_(other.lattice_vectors_), metric_(other.metric_) 
        {
            update_metric();
        }
        
        // Access lattice vectors
        const array3d_flt64& operator[](size_t i) const { return lattice_vectors_[i]; }
        array3d_flt64& operator[](size_t i) { return lattice_vectors_[i]; }

        // get lattice metric
        void update_metric() {
            metric_ = sym_matrix3d_flt64(lattice_vectors_[0].dot(lattice_vectors_[0]),
                                  lattice_vectors_[1].dot(lattice_vectors_[1]),
                                  lattice_vectors_[2].dot(lattice_vectors_[2]),
                                  lattice_vectors_[0].dot(lattice_vectors_[1]),
                                  lattice_vectors_[0].dot(lattice_vectors_[2]),
                                  lattice_vectors_[1].dot(lattice_vectors_[2]));
        }

        // get volume of cell
        double get_volume() const {
            return std::abs(lattice_vectors_.det());
        }

        // normalize lattice vectors to unit length
        matrix3d_flt64 normalize() const {
            matrix3d_flt64 result;
            for(int i = 0; i < 3; i++) {
                result[i] = lattice_vectors_[i] / lattice_vectors_[i].norm();
            }
            return result;
        }

        // get inter-planar spacing
        array3d_flt64 get_interplanar_spacing() const {
            double volume = get_volume();
            array3d_flt64 b1 = lattice_vectors_[1].cross(lattice_vectors_[2]) / volume;
            array3d_flt64 b2 = lattice_vectors_[2].cross(lattice_vectors_[0]) / volume;
            array3d_flt64 b3 = lattice_vectors_[0].cross(lattice_vectors_[1]) / volume;
            return array3d_flt64(1.0 / b1.norm(), 1.0 / b2.norm(), 1.0 / b3.norm());
        }

        array3d_flt64 get_cell_lengths() const {
            return array3d_flt64(lattice_vectors_[0].norm(), lattice_vectors_[1].norm(), lattice_vectors_[2].norm());
        }

        double calculate_distance_squared(const point_flt64& p1, const point_flt64& p2, const array3d_flt64& cell_shift, array3d_flt64& difference) const {
            difference = array3d_flt64(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
            difference += cell_shift;
            return difference.dot(metric_ * difference);
        }

        double calculate_distance_squared(const array3d_flt64& difference) const {
            return difference.dot(metric_ * difference);
        }

        double calculate_distance(const point_flt64& p1, const point_flt64& p2, const array3d_flt64& cell_shift, array3d_flt64& difference) const {
            return std::sqrt(calculate_distance_squared(p1, p2, cell_shift, difference));
        }

        array3d_flt64 fractional_to_cartesian(const array3d_flt64& fractional) const {
            return lattice_vectors_.transpose_mult(fractional);
        }

        void dump() const;

    private:
        matrix3d_flt64 lattice_vectors_;
        // matrix3d_flt64 inverse_lattice_vectors_;
        sym_matrix3d_flt64 metric_;
    };

    // Convert cell parameters to lattice vectors
    std::unique_ptr<lattice_t> cell_info_to_lattice(const array3d_flt64& cell_lengths, const array3d_flt64& cell_angles);

}}