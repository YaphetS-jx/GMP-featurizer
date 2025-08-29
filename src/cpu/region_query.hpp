#pragma once
#include "containers.hpp"
#include "atom.hpp"
#include "math.hpp"
#include "tree.hpp"
#include "morton_codes.hpp"
#include "util.hpp"

namespace gmp { namespace region_query {

    using namespace gmp::atom;
    using namespace gmp::geometry;
    using namespace gmp::tree;
    using namespace gmp::containers;
    using namespace gmp::math;

    template <typename FloatType>
    struct query_result_t {
        array3d_t<FloatType> difference;
        FloatType distance_squared;
        int neighbor_index;
    };

    template <typename MortonCodeType, typename FloatType, typename IndexType>
    class check_sphere_t : public compare_op_t<MortonCodeType> {
    private:
        point3d_t<FloatType> position;
        FloatType radius;
        FloatType radius2;
        array3d_t<FloatType> frac_radius;
        array3d_t<IndexType> cell_shift_start;
        array3d_t<IndexType> cell_shift_end;
        const IndexType num_bits_per_dim;
        const array3d_bool periodicity;
        const lattice_t<FloatType>* lattice;

    public:
        explicit check_sphere_t(const IndexType num_bits_per_dim, const array3d_bool periodicity, const lattice_t<FloatType>* lattice);

        void update_point_radius(point3d_t<FloatType> position_in, FloatType radius);
        array3d_t<IndexType> get_cell_shift_start() const;
        array3d_t<IndexType> get_cell_shift_end() const;

        bool operator()(MortonCodeType lower_bound, MortonCodeType upper_bound) const override;
        std::vector<array3d_t<IndexType>> operator()(MortonCodeType morton_code) const override;
    };

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    class region_query_t {
    public:
        region_query_t(const unit_cell_t<FloatType>* unit_cell, const uint8_t num_bits_per_dim = 10);
        ~region_query_t() = default;
    
    private: 
        vector<MortonCodeType> unique_morton_codes;
        vector<IndexType> offsets;
        vector<IndexType> sorted_indexes;
        const int num_bits_per_dim;

    private:
        void get_morton_codes(const vector<atom_t<FloatType>>& atoms, const uint8_t num_bits_per_dim = 10);

    public:         
        const vector<MortonCodeType>& get_unique_morton_codes() const;
        const vector<IndexType>& get_offsets() const;
        const vector<IndexType>& get_sorted_indexes() const;
        using result_t = std::vector<query_result_t<FloatType>>;
        using sphere_op_t = check_sphere_t<MortonCodeType, FloatType, IndexType>;

        result_t query(const point3d_t<FloatType>& position, const FloatType cutoff, 
            const binary_radix_tree_t<MortonCodeType, IndexType>* brt, const unit_cell_t<FloatType>* unit_cell);
    };
}}