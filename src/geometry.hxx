#pragma once
#include <cmath>
#include <memory>
#include <iostream>
#include "geometry.hpp"

namespace gmp { namespace geometry {

    // Stream insertion operator for point3d_t
    template <typename T>
    inline std::ostream& operator<<(std::ostream& os, const point3d_t<T>& p) {
        os << "(" << p.x << ", " << p.y << ", " << p.z << ")";
        return os;
    }

    // lattice_t implementations
    template <typename T>
    inline lattice_t<T>::lattice_t(matrix3d_type lattice_vectors) : lattice_vectors_(lattice_vectors), metric_{} 
    {
        update_metric();
    }
    
    // copy constructor
    template <typename T>
    inline lattice_t<T>::lattice_t(const lattice_t& other) : lattice_vectors_(other.lattice_vectors_), metric_(other.metric_) 
    {
        update_metric();
    }
    
    // Access lattice vectors
    template <typename T>
    inline const typename lattice_t<T>::array3d_type& lattice_t<T>::operator[](size_t i) const { 
        return lattice_vectors_[i]; 
    }
    
    template <typename T>
    inline typename lattice_t<T>::array3d_type& lattice_t<T>::operator[](size_t i) { 
        return lattice_vectors_[i]; 
    }

    // get lattice metric
    template <typename T>
    inline void lattice_t<T>::update_metric() {
        metric_ = sym_matrix3d_type(lattice_vectors_[0].dot(lattice_vectors_[0]),
                              lattice_vectors_[1].dot(lattice_vectors_[1]),
                              lattice_vectors_[2].dot(lattice_vectors_[2]),
                              lattice_vectors_[0].dot(lattice_vectors_[1]),
                              lattice_vectors_[0].dot(lattice_vectors_[2]),
                              lattice_vectors_[1].dot(lattice_vectors_[2]));
    }

    // get volume of cell
    template <typename T>
    inline T lattice_t<T>::get_volume() const {
        return std::abs(lattice_vectors_.det());
    }

    // normalize lattice vectors to unit length
    template <typename T>
    inline typename lattice_t<T>::matrix3d_type lattice_t<T>::normalize() const {
        matrix3d_type result;
        for(int i = 0; i < 3; i++) {
            result[i] = lattice_vectors_[i] / lattice_vectors_[i].norm();
        }
        return result;
    }

    // get inter-planar spacing
    template <typename T>
    inline typename lattice_t<T>::array3d_type lattice_t<T>::get_interplanar_spacing() const {
        T volume = get_volume();
        array3d_type b1 = lattice_vectors_[1].cross(lattice_vectors_[2]) / volume;
        array3d_type b2 = lattice_vectors_[2].cross(lattice_vectors_[0]) / volume;
        array3d_type b3 = lattice_vectors_[0].cross(lattice_vectors_[1]) / volume;
        return array3d_type(static_cast<T>(1.0) / b1.norm(), static_cast<T>(1.0) / b2.norm(), static_cast<T>(1.0) / b3.norm());
    }

    template <typename T>
    inline typename lattice_t<T>::array3d_type lattice_t<T>::get_cell_lengths() const {
        return array3d_type(lattice_vectors_[0].norm(), lattice_vectors_[1].norm(), lattice_vectors_[2].norm());
    }

    template <typename T>
    inline T lattice_t<T>::calculate_distance_squared(const point_type& p1, const point_type& p2, const array3d_type& cell_shift, array3d_type& difference) const {
        difference = array3d_type(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
        difference += cell_shift;
        return difference.dot(metric_ * difference);
    }

    template <typename T>
    inline T lattice_t<T>::calculate_distance_squared(const array3d_type& difference) const {
        return difference.dot(metric_ * difference);
    }

    template <typename T>
    inline T lattice_t<T>::calculate_distance(const point_type& p1, const point_type& p2, const array3d_type& cell_shift, array3d_type& difference) const {
        return std::sqrt(calculate_distance_squared(p1, p2, cell_shift, difference));
    }

    template <typename T>
    inline typename lattice_t<T>::array3d_type lattice_t<T>::fractional_to_cartesian(const array3d_type& fractional) const {
        return lattice_vectors_.transpose_mult(fractional);
    }

    template <typename T>
    inline void lattice_t<T>::dump() const {
        std::cout << "lattice_vectors_: " << std::endl;
        std::cout << lattice_vectors_[0][0] << " " << lattice_vectors_[0][1] << " " << lattice_vectors_[0][2] << std::endl;
        std::cout << lattice_vectors_[1][0] << " " << lattice_vectors_[1][1] << " " << lattice_vectors_[1][2] << std::endl;
        std::cout << lattice_vectors_[2][0] << " " << lattice_vectors_[2][1] << " " << lattice_vectors_[2][2] << std::endl;
        std::cout << "metric_: " << std::endl;
        std::cout << metric_[0] << " " << metric_[1] << " " << metric_[2] << std::endl;
        std::cout << metric_[3] << " " << metric_[4] << " " << metric_[5] << std::endl;
    }

    // Convert cell parameters to lattice vectors
    template <typename T>
    inline std::unique_ptr<lattice_t<T>> cell_info_to_lattice(const array3d_t<T>& cell_lengths, const array3d_t<T>& cell_angles) {
        
        // Define rotated X, Y, Z system with default orientation
        array3d_t<T> Z{static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0)};  // ab_normal
        array3d_t<T> X{static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0)};  // default a_direction
        array3d_t<T> Y = Z.cross(X);    // Y is perpendicular to both Z and X
        
        // Tolerance for angles near 90°
        T eps = static_cast<T>(2) * std::numeric_limits<T>::epsilon() * static_cast<T>(90.0);
        T cos_alpha, cos_beta, cos_gamma, sin_gamma;
        
        if (std::abs(std::abs(cell_angles[0]) - static_cast<T>(90.0)) < eps)
            cos_alpha = static_cast<T>(0.0);
        else
            cos_alpha = std::cos(cell_angles[0] * M_PI / static_cast<T>(180.0));

        if (std::abs(std::abs(cell_angles[1]) - static_cast<T>(90.0)) < eps)
            cos_beta = static_cast<T>(0.0);
        else
            cos_beta = std::cos(cell_angles[1] * M_PI / static_cast<T>(180.0));

        if (std::abs(cell_angles[2] - static_cast<T>(90.0)) < eps) {
            cos_gamma = static_cast<T>(0.0);
            sin_gamma = static_cast<T>(1.0);
        } else if (std::abs(cell_angles[2] + static_cast<T>(90.0)) < eps) {
            cos_gamma = static_cast<T>(0.0);
            sin_gamma = static_cast<T>(-1.0);
        } else {
            cos_gamma = std::cos(cell_angles[2] * M_PI / static_cast<T>(180.0));
            sin_gamma = std::sin(cell_angles[2] * M_PI / static_cast<T>(180.0));
        }

        // Build cell vectors in intermediate (X,Y,Z) coordinate system
        array3d_t<T> va{cell_lengths[0], static_cast<T>(0.0), static_cast<T>(0.0)};
        array3d_t<T> vb{cell_lengths[1] * cos_gamma, cell_lengths[1] * sin_gamma, static_cast<T>(0.0)};
        
        T cx = cos_beta;
        T cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
        T cz = std::sqrt(static_cast<T>(1.0) - cx * cx - cy * cy);
        array3d_t<T> vc{cell_lengths[2] * cx, cell_lengths[2] * cy, cell_lengths[2] * cz};

        // Build transformation matrix T from basis vectors X, Y, Z
        matrix3d_t<T> T_matrix(X, Y, Z);

        // Build intermediate matrix with rows va, vb, vc
        matrix3d_t<T> abc(va, vb, vc);

        // Multiply abc and T to obtain final cell matrix        
        return std::make_unique<lattice_t<T>>(abc * T_matrix);
    }

}}