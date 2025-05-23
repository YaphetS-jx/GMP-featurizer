#pragma once
#include "types.hpp"
#include "atom.hpp"
#include "math.hpp"

namespace gmp { namespace query {

    using namespace gmp::containers;
    using namespace gmp::atom;

    /* Use uniform-grid / hash build method. */

    struct query_result_t {
        array3d_flt64 difference;
        double distance_squared;
        int neighbor_index;

        query_result_t(array3d_flt64 difference_, double distance_squared_, int neighbor_index_) 
            : difference(difference_), distance_squared(distance_squared_), neighbor_index(neighbor_index_) {}
    };

    class query_info_t {
    public:
        query_info_t(const unit_cell_t* unit_cell, const double cutoff);
        ~query_info_t() = default;

        array3d_int32 get_bin_index_3d(const point_flt64& position) const;

        int get_bin_index_1d(const point_flt64& position) const;

        int get_bin_index_1d(const array3d_int32& bin_index) const;

        vec<query_result_t> get_neighbor_list(const double cutoff, const point_flt64& position, const unit_cell_t* unit_cell) const;
        
    private:
        array3d_int32 num_bins_;
        vec<int> bin_atoms_;
        vec<int> bin_offset_;
        array3d_int8 bin_ranges_;
    };
}}