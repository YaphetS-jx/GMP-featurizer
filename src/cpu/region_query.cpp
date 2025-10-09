#include "region_query.hpp"
#include <algorithm>
#include <cmath>
#include "gmp_float.hpp"

namespace gmp { namespace region_query {

    // check_sphere_t implementations
    template <typename FloatType, typename IndexType>
    check_sphere_t<FloatType, IndexType>::check_sphere_t(
        const IndexType num_bits_per_dim, const array3d_bool periodicity, const lattice_t<FloatType>* lattice)
        : num_bits_per_dim(num_bits_per_dim), periodicity(periodicity), lattice(lattice) 
    {}

    template <typename FloatType, typename IndexType>
    void check_sphere_t<FloatType, IndexType>::update_point_radius(
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

    template <typename FloatType, typename IndexType>
    array3d_t<IndexType> check_sphere_t<FloatType, IndexType>::get_cell_shift_start() const {
        return cell_shift_start;
    }

    template <typename FloatType, typename IndexType>
    array3d_t<IndexType> check_sphere_t<FloatType, IndexType>::get_cell_shift_end() const {
        return cell_shift_end;
    }

    template <typename FloatType, typename IndexType>
    bool check_sphere_t<FloatType, IndexType>::operator()(
        const array3d_t<FloatType>& lower_coords, const array3d_t<FloatType>& upper_coords) const 
    {
        
        auto get_difference = [](FloatType min, FloatType max, FloatType point) {
            return (min <= point && point <= max) ? 0 : (point < min) ? min - point : point - max;
        };

        for (auto shift_z = cell_shift_start[2]; shift_z <= cell_shift_end[2]; shift_z++) {
            for (auto shift_y = cell_shift_start[1]; shift_y <= cell_shift_end[1]; shift_y++) {
                for (auto shift_x = cell_shift_start[0]; shift_x <= cell_shift_end[0]; shift_x++) {
                    auto x_min_shift = lower_coords[0] + shift_x;
                    auto y_min_shift = lower_coords[1] + shift_y;
                    auto z_min_shift = lower_coords[2] + shift_z;
                    auto x_max_shift = upper_coords[0] + shift_x;
                    auto y_max_shift = upper_coords[1] + shift_y;
                    auto z_max_shift = upper_coords[2] + shift_z;

                    array3d_t<FloatType> difference = {
                        get_difference(x_min_shift, x_max_shift, position.x), 
                        get_difference(y_min_shift, y_max_shift, position.y),
                        get_difference(z_min_shift, z_max_shift, position.z)
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

    template <typename FloatType, typename IndexType>
    std::vector<array3d_t<IndexType>> check_sphere_t<FloatType, IndexType>::operator()(
        const array3d_t<FloatType>& lower_coords, FloatType size_per_dim) const 
    {
        auto get_difference = [](FloatType min, FloatType max, FloatType point) {
            return (min <= point && point <= max) ? 0 : (point < min) ? min - point : point - max;
        };

        std::vector<array3d_t<IndexType>> result;            

        for (auto shift_z = cell_shift_start[2]; shift_z <= cell_shift_end[2]; shift_z++) {
            for (auto shift_y = cell_shift_start[1]; shift_y <= cell_shift_end[1]; shift_y++) {
                for (auto shift_x = cell_shift_start[0]; shift_x <= cell_shift_end[0]; shift_x++) {
                    auto x_min_shift = lower_coords[0] + shift_x;
                    auto y_min_shift = lower_coords[1] + shift_y;
                    auto z_min_shift = lower_coords[2] + shift_z;
                    auto x_max_shift = x_min_shift + size_per_dim;
                    auto y_max_shift = y_min_shift + size_per_dim;
                    auto z_max_shift = z_min_shift + size_per_dim;

                    array3d_t<FloatType> difference = {
                        get_difference(x_min_shift, x_max_shift, position.x), 
                        get_difference(y_min_shift, y_max_shift, position.y),
                        get_difference(z_min_shift, z_max_shift, position.z)
                    };
                    auto distance_squared = lattice->calculate_distance_squared(difference);
                    if (distance_squared <= radius2) {
                        array3d_t<IndexType> shift;
                        shift[0] = shift_x;
                        shift[1] = shift_y;
                        shift[2] = shift_z;
                        result.emplace_back(shift);
                    }
                }
            }
        }
        return result;
    }

    template class check_sphere_t<gmp::gmp_float, int32_t>;

    // region_query_t implementations
    template <typename MortonCodeType, typename IndexType, typename FloatType>
    region_query_t<MortonCodeType, IndexType, FloatType>::region_query_t(
        const unit_cell_t<FloatType>* unit_cell, const uint8_t num_bits_per_dim) 
        : num_bits_per_dim(num_bits_per_dim) {
        // get morton codes
        get_morton_codes(unit_cell->get_atoms(), num_bits_per_dim);
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    void region_query_t<MortonCodeType, IndexType, FloatType>::get_morton_codes(
        const vector<atom_t<FloatType>>& atoms, const uint8_t num_bits_per_dim) {
        vector<MortonCodeType> morton_codes;
        auto natom = atoms.size();
        morton_codes.reserve(natom);
        for (const auto& atom : atoms) {
            MortonCodeType mc_x = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.pos.x, num_bits_per_dim);
            MortonCodeType mc_y = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.pos.y, num_bits_per_dim);
            MortonCodeType mc_z = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.pos.z, num_bits_per_dim);
            morton_codes.push_back(interleave_bits(mc_x, mc_y, mc_z, num_bits_per_dim));
        }

        // index mapping from morton codes to atoms
        sorted_indexes = util::sort_indexes<MortonCodeType, IndexType, vector>(morton_codes);
        std::sort(morton_codes.begin(), morton_codes.end());

        // compact
        unique_morton_codes.clear();
        unique_morton_codes.reserve(natom);
        std::vector<IndexType> indexing(natom+1);
        for (auto i = 0; i < natom+1; i++) indexing[i] = i;
        // get same flag 
        std::vector<bool> same(natom);
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

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    const vector<MortonCodeType>& region_query_t<MortonCodeType, IndexType, FloatType>::get_unique_morton_codes() const {
        return unique_morton_codes;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    const vector<IndexType>& region_query_t<MortonCodeType, IndexType, FloatType>::get_offsets() const {
        return offsets;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    const vector<IndexType>& region_query_t<MortonCodeType, IndexType, FloatType>::get_sorted_indexes() const {
        return sorted_indexes;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    typename region_query_t<MortonCodeType, IndexType, FloatType>::result_t 
    region_query_t<MortonCodeType, IndexType, FloatType>::query(
        const point3d_t<FloatType>& position, const FloatType cutoff, 
        const binary_radix_tree_t<IndexType, FloatType>* brt, const unit_cell_t<FloatType>* unit_cell) {
        sphere_op_t compare_op(num_bits_per_dim, unit_cell->get_periodicity(), unit_cell->get_lattice());
        compare_op.update_point_radius(position, cutoff);
        auto cutoff_squared = cutoff * cutoff;
        auto query_mc = brt->traverse(compare_op);

        result_t result;
        for (const auto& [index, shifts] : query_mc) {
            for (const auto& shift : shifts) {
                for (auto idx = offsets[index]; idx < offsets[index+1]; idx++) {
                    auto atom_index = sorted_indexes[idx];
                    auto atom_position = unit_cell->get_atoms()[atom_index].pos;
                    array3d_t<FloatType> difference;
                    auto distance2 = unit_cell->get_lattice()->calculate_distance_squared(
                        atom_position, position, array3d_t<FloatType>{static_cast<FloatType>(shift[0]), 
                        static_cast<FloatType>(shift[1]), static_cast<FloatType>(shift[2])}, difference);
                    if (distance2 < cutoff_squared) {
                        array3d_t<FloatType> difference_cartesian = unit_cell->get_lattice()->fractional_to_cartesian(difference);
                        result.push_back({difference_cartesian, distance2, atom_index});
                    }
                }
            }
        }

        std::sort(result.begin(), result.end(), 
        [](const query_result_t<FloatType>& a, const query_result_t<FloatType>& b) {
            return a.distance_squared < b.distance_squared || 
            (a.distance_squared == b.distance_squared && a.neighbor_index < b.neighbor_index);
        });
        return result;
    }
    
    template class region_query_t<uint32_t, int32_t, gmp::gmp_float>;
}} 