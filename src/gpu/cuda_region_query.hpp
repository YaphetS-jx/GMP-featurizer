#pragma once
#include "containers.hpp"
#include "atom.hpp"
#include "math.hpp"
#include "cuda_tree.hpp"
#include "morton_codes.hpp"
#include "util.hpp"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

namespace gmp { namespace region_query {

    using namespace gmp::atom;
    using namespace gmp::geometry;
    using namespace gmp::tree;
    using namespace gmp::containers;
    using namespace gmp::math;

    template <typename FloatType>
    struct cuda_query_result_t {
        array3d_t<FloatType> difference;
        FloatType distance_squared;
        int neighbor_index;

        // Default constructor for trivially copyable requirement
        cuda_query_result_t() = default;
        
        cuda_query_result_t(array3d_t<FloatType> difference_, FloatType distance_squared_, int neighbor_index_)
            : difference(difference_), distance_squared(distance_squared_), neighbor_index(neighbor_index_) {}

        // Add comparison operator for sorting
        bool operator<(const cuda_query_result_t& other) const {
            if (distance_squared != other.distance_squared) {
                return distance_squared < other.distance_squared;
            }
            if (neighbor_index != other.neighbor_index) {
                return neighbor_index < other.neighbor_index;
            }
            return true;
        }
    };

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType = vector_device<array3d_int32>>
    class cuda_check_sphere_t {
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
        explicit cuda_check_sphere_t(const IndexType num_bits_per_dim, const array3d_bool periodicity, const lattice_t<FloatType>* lattice);

        void update_point_radius(point3d_t<FloatType> position_in, FloatType radius);
        array3d_t<IndexType> get_cell_shift_start() const;
        array3d_t<IndexType> get_cell_shift_end() const;


    };

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    class cuda_region_query_t {
    public:
        cuda_region_query_t(const unit_cell_t<FloatType>* unit_cell, const uint8_t num_bits_per_dim = 10, cudaStream_t stream = 0);
        ~cuda_region_query_t() = default;
    
    private: 
        vector_device<MortonCodeType> unique_morton_codes;
        vector_device<IndexType> offsets;
        vector_device<IndexType> sorted_indexes;
        vector_device<point3d_t<FloatType>> atom_positions;
        std::unique_ptr<cuda_binary_radix_tree_t<MortonCodeType, IndexType>> brt;
        const int num_bits_per_dim;
        const unit_cell_t<FloatType>* unit_cell_ptr;

    private:
        void get_morton_codes(const vector<atom_t<FloatType>>& atoms, const uint8_t num_bits_per_dim, cudaStream_t stream);

    public:         
        const vector_device<MortonCodeType>& get_unique_morton_codes() const;
        const vector_device<IndexType>& get_offsets() const;
        const vector_device<IndexType>& get_sorted_indexes() const;
        using result_t = vector_device<int>; // For now, just return indices
        using sphere_op_t = cuda_check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>;

        result_t query(const point3d_t<FloatType>& position, const FloatType cutoff, 
                      const unit_cell_t<FloatType>* unit_cell, cudaStream_t stream = 0);
    };

    // Explicit instantiations
    template struct cuda_query_result_t<float>;
    template struct cuda_query_result_t<double>;
    
    template class cuda_check_sphere_t<uint32_t, float, int32_t, vector_device<array3d_int32>>;
    template class cuda_check_sphere_t<uint32_t, double, int32_t, vector_device<array3d_int32>>;
    
    template class cuda_region_query_t<uint32_t, int32_t, float, vector_device<array3d_int32>>;
    template class cuda_region_query_t<uint32_t, int32_t, double, vector_device<array3d_int32>>;

}} // namespace gmp::region_query 