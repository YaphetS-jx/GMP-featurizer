#include "region_query.hpp"
#include <algorithm>
#include <cmath>

namespace gmp { namespace region_query {

    // query_result_t implementations
    template <typename FloatType>
    query_result_t<FloatType>::query_result_t(array3d_t<FloatType> difference_, FloatType distance_squared_, int neighbor_index_) 
        : difference(difference_), distance_squared(distance_squared_), neighbor_index(neighbor_index_) {}

    template <typename FloatType>
    bool query_result_t<FloatType>::operator<(const query_result_t& other) const {
        if (distance_squared != other.distance_squared) {
            return distance_squared < other.distance_squared;
        }
        if (neighbor_index != other.neighbor_index) {
            return neighbor_index < other.neighbor_index;
        }
        return true;
    }

    template struct query_result_t<float>;
    template struct query_result_t<double>;

    // check_sphere_t implementations
    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::check_sphere_t(
        const IndexType num_bits_per_dim, const array3d_bool periodicity, const lattice_t<FloatType>* lattice)
        : num_bits_per_dim(num_bits_per_dim), periodicity(periodicity), lattice(lattice) 
    {}

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    void check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::update_point_radius(
        point3d_t<FloatType> position_in, FloatType radius) {
        // get fractional radius 
        radius2 = radius * radius;
        frac_radius = radius / lattice->get_cell_lengths();
        position = position_in;

        // Initialize cell shifts
        cell_shift_start[0] = std::floor(position.x - frac_radius[0]);
        cell_shift_end[0] = std::floor(position.x + frac_radius[0]);
        cell_shift_start[1] = std::floor(position.y - frac_radius[1]);
        cell_shift_end[1] = std::floor(position.y + frac_radius[1]);
        cell_shift_start[2] = std::floor(position.z - frac_radius[2]);
        cell_shift_end[2] = std::floor(position.z + frac_radius[2]);
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    array3d_t<IndexType> check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::get_cell_shift_start() const {
        return cell_shift_start;
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    array3d_t<IndexType> check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::get_cell_shift_end() const {
        return cell_shift_end;
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    bool check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::operator()(
        MortonCodeType lower_bound, MortonCodeType upper_bound) const {
        MortonCodeType x_min, y_min, z_min;
        deinterleave_bits(lower_bound, num_bits_per_dim, x_min, y_min, z_min);
        MortonCodeType x_max, y_max, z_max;
        deinterleave_bits(upper_bound, num_bits_per_dim, x_max, y_max, z_max);

        auto x_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim);
        auto y_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim);
        auto z_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim);

        FloatType size_per_dim = 1.0 / (1 << (num_bits_per_dim - 1));
        auto x_max_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_max, num_bits_per_dim) + size_per_dim;
        auto y_max_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_max, num_bits_per_dim) + size_per_dim;
        auto z_max_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_max, num_bits_per_dim) + size_per_dim;

        auto point_x = position.x;
        auto point_y = position.y;
        auto point_z = position.z;

        auto get_difference = [](FloatType min, FloatType max, FloatType point) {
            return (min <= point && point <= max) ? 0 : (point < min) ? min - point : point - max;
        };

        for (auto shift_z = cell_shift_start[2]; shift_z <= cell_shift_end[2]; shift_z++) {
            for (auto shift_y = cell_shift_start[1]; shift_y <= cell_shift_end[1]; shift_y++) {
                for (auto shift_x = cell_shift_start[0]; shift_x <= cell_shift_end[0]; shift_x++) {
                    auto x_min_shift = x_min_f + shift_x;
                    auto y_min_shift = y_min_f + shift_y;
                    auto z_min_shift = z_min_f + shift_z;
                    auto x_max_shift = x_max_f + shift_x;
                    auto y_max_shift = y_max_f + shift_y;
                    auto z_max_shift = z_max_f + shift_z;

                    array3d_t<FloatType> difference = {
                        get_difference(x_min_shift, x_max_shift, point_x), 
                        get_difference(y_min_shift, y_max_shift, point_y),
                        get_difference(z_min_shift, z_max_shift, point_z)
                    };
                    auto distance_squared = lattice->calculate_distance_squared(difference);                        
                    if (distance_squared <= radius2) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    VecType check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::operator()(MortonCodeType morton_code) const {
        MortonCodeType x_min, y_min, z_min;
        deinterleave_bits(morton_code, num_bits_per_dim, x_min, y_min, z_min);
        
        FloatType size_per_dim = 1.0 / (1 << (num_bits_per_dim - 1));
        auto x_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim);
        auto y_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim);
        auto z_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim);
        auto x_max_f = x_min_f + size_per_dim;
        auto y_max_f = y_min_f + size_per_dim;
        auto z_max_f = z_min_f + size_per_dim;

        auto point_x = position.x;
        auto point_y = position.y;
        auto point_z = position.z;

        auto get_difference = [](FloatType min, FloatType max, FloatType point) {
            return (min <= point && point <= max) ? 0 : (point < min) ? min - point : point - max;
        };

        VecType result;            

        for (auto shift_z = cell_shift_start[2]; shift_z <= cell_shift_end[2]; shift_z++) {
            for (auto shift_y = cell_shift_start[1]; shift_y <= cell_shift_end[1]; shift_y++) {
                for (auto shift_x = cell_shift_start[0]; shift_x <= cell_shift_end[0]; shift_x++) {
                    auto x_min_shift = x_min_f + shift_x;
                    auto y_min_shift = y_min_f + shift_y;
                    auto z_min_shift = z_min_f + shift_z;
                    auto x_max_shift = x_max_f + shift_x;
                    auto y_max_shift = y_max_f + shift_y;
                    auto z_max_shift = z_max_f + shift_z;

                    array3d_t<FloatType> difference = {
                        get_difference(x_min_shift, x_max_shift, point_x), 
                        get_difference(y_min_shift, y_max_shift, point_y),
                        get_difference(z_min_shift, z_max_shift, point_z)
                    };
                    auto distance_squared = lattice->calculate_distance_squared(difference);
                    if (distance_squared <= radius2) {
                        result.emplace_back(shift_x, shift_y, shift_z);
                    }
                }
            }
        }
        return result;
    }

    template class check_sphere_t<uint32_t, float, int32_t, vector<array3d_int32>>;
    template class check_sphere_t<uint32_t, double, int32_t, vector<array3d_int32>>;

    // region_query_t implementations
    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    region_query_t<MortonCodeType, IndexType, FloatType, VecType>::region_query_t(
        const unit_cell_t<FloatType>* unit_cell, const uint8_t num_bits_per_dim) 
        : num_bits_per_dim(num_bits_per_dim) {
        // get morton codes
        get_morton_codes(unit_cell->get_atoms(), num_bits_per_dim);
        
        // build tree
        brt = std::make_unique<binary_radix_tree_t<MortonCodeType, IndexType>>(unique_morton_codes, num_bits_per_dim * 3);
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    void region_query_t<MortonCodeType, IndexType, FloatType, VecType>::get_morton_codes(
        const vector<atom_t<FloatType>>& atoms, const uint8_t num_bits_per_dim) {
        vector<MortonCodeType> morton_codes;
        auto natom = atoms.size();
        morton_codes.reserve(natom);
        for (const auto& atom : atoms) {
            MortonCodeType mc_x = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.x(), num_bits_per_dim);
            MortonCodeType mc_y = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.y(), num_bits_per_dim);
            MortonCodeType mc_z = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.z(), num_bits_per_dim);
            morton_codes.push_back(interleave_bits(mc_x, mc_y, mc_z, num_bits_per_dim));
        }

        // index mapping from morton codes to atoms
        sorted_indexes = util::sort_indexes<MortonCodeType, IndexType, vector>(morton_codes);
        std::sort(morton_codes.begin(), morton_codes.end());

        // compact
        unique_morton_codes.clear();
        unique_morton_codes.reserve(natom);
        vector<IndexType> indexing(natom+1);
        for (auto i = 0; i < natom+1; i++) indexing[i] = i;
        // get same flag 
        vector<bool> same(natom);
        same[0] = true;
        for (auto i = 1; i < natom; i++) {
            same[i] = morton_codes[i] != morton_codes[i-1];
        }
        // get unique morton codes
        for (auto i = 0; i < natom; i++) {
            if (same[i]) {
                unique_morton_codes.push_back(morton_codes[i]);
            }
        }

        // get offsets
        offsets.clear();
        offsets.reserve(natom + 1);
        for (auto i = 0; i < natom; i++) {
            if (same[i]) {
                offsets.push_back(indexing[i]);
            }
        }
        offsets.push_back(natom);
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    const vector<MortonCodeType>& region_query_t<MortonCodeType, IndexType, FloatType, VecType>::get_unique_morton_codes() const {
        return unique_morton_codes;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    const vector<IndexType>& region_query_t<MortonCodeType, IndexType, FloatType, VecType>::get_offsets() const {
        return offsets;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    const vector<IndexType>& region_query_t<MortonCodeType, IndexType, FloatType, VecType>::get_sorted_indexes() const {
        return sorted_indexes;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    typename region_query_t<MortonCodeType, IndexType, FloatType, VecType>::result_t 
    region_query_t<MortonCodeType, IndexType, FloatType, VecType>::query(
        const point3d_t<FloatType>& position, const FloatType cutoff, const unit_cell_t<FloatType>* unit_cell) {
        sphere_op_t compare_op(num_bits_per_dim, unit_cell->get_periodicity(), unit_cell->get_lattice());
        compare_op.update_point_radius(position, cutoff);
        auto cutoff_squared = cutoff * cutoff;
        auto query_mc = brt->traverse(compare_op);

        result_t result;
        for (const auto& [index, shifts] : query_mc) {
            for (const auto& shift : shifts) {
                for (auto idx = offsets[index]; idx < offsets[index+1]; idx++) {
                    auto atom_index = sorted_indexes[idx];
                    auto atom_position = unit_cell->get_atoms()[atom_index].pos();
                    array3d_t<FloatType> difference;
                    auto distance2 = unit_cell->get_lattice()->calculate_distance_squared(
                        atom_position, position, array3d_t<FloatType>{static_cast<FloatType>(shift[0]), 
                        static_cast<FloatType>(shift[1]), static_cast<FloatType>(shift[2])}, difference);
                    if (distance2 < cutoff_squared) {
                        array3d_t<FloatType> difference_cartesian = unit_cell->get_lattice()->fractional_to_cartesian(difference);
                        result.emplace_back(difference_cartesian, distance2, atom_index);
                    }
                }
            }
        }

        std::sort(result.begin(), result.end());
        return result;
    }
    
    template class region_query_t<uint32_t, int32_t, float, vector<array3d_int32>>;
    template class region_query_t<uint32_t, int32_t, double, vector<array3d_int32>>;

}} 