#include "geometry.hpp"

namespace gmp { namespace geometry {

    using namespace gmp::math;

    // Convert cell parameters to lattice vectors
    lattice_t cell_info_to_lattice(const array3d_flt64& cell_lengths, const array3d_flt64& cell_angles) {
        
        // Define rotated X, Y, Z system with default orientation
        array3d_flt64 Z{0.0, 0.0, 1.0};  // ab_normal
        array3d_flt64 X{1.0, 0.0, 0.0};  // default a_direction
        array3d_flt64 Y = Z.cross(X);    // Y is perpendicular to both Z and X
        
        // Tolerance for angles near 90°
        double eps = 2 * std::numeric_limits<double>::epsilon() * 90.0;
        double cos_alpha, cos_beta, cos_gamma, sin_gamma;
        
        if (std::abs(std::abs(cell_angles[0]) - 90.0) < eps)
            cos_alpha = 0.0;
        else
            cos_alpha = std::cos(cell_angles[0] * M_PI / 180.0);

        if (std::abs(std::abs(cell_angles[1]) - 90.0) < eps)
            cos_beta = 0.0;
        else
            cos_beta = std::cos(cell_angles[1] * M_PI / 180.0);

        if (std::abs(cell_angles[2] - 90.0) < eps) {
            cos_gamma = 0.0;
            sin_gamma = 1.0;
        } else if (std::abs(cell_angles[2] + 90.0) < eps) {
            cos_gamma = 0.0;
            sin_gamma = -1.0;
        } else {
            cos_gamma = std::cos(cell_angles[2] * M_PI / 180.0);
            sin_gamma = std::sin(cell_angles[2] * M_PI / 180.0);
        }

        // Build cell vectors in intermediate (X,Y,Z) coordinate system
        array3d_flt64 va{cell_lengths[0], 0.0, 0.0};
        array3d_flt64 vb{cell_lengths[1] * cos_gamma, cell_lengths[1] * sin_gamma, 0.0};
        
        double cx = cos_beta;
        double cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
        double cz = std::sqrt(1.0 - cx * cx - cy * cy);
        array3d_flt64 vc{cell_lengths[2] * cx, cell_lengths[2] * cy, cell_lengths[2] * cz};

        // Build transformation matrix T from basis vectors X, Y, Z
        matrix3d_flt64 T(X, Y, Z);

        // Build intermediate matrix with rows va, vb, vc
        matrix3d_flt64 abc(va, vb, vc);

        // Multiply abc and T to obtain final cell matrix
        return lattice_t(abc * T);
    }
}}