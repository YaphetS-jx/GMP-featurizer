#pragma once
#include "containers.hpp"
#include "atom.hpp"
#include "math.hpp"
#include "cuda_tree.hpp"
#include "morton_codes.hpp"
#include "util.hpp"

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
    };

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    class cuda_region_query_t {
    public:
        cuda_region_query_t(
            const vector<MortonCodeType>& h_morton_codes, const IndexType num_bits_per_dim, 
            const vector<IndexType>& h_offsets, const vector<IndexType>& h_sorted_indexes, 
            cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream());
        ~cuda_region_query_t() = default;
    
        vector_device<IndexType> offsets;
        vector_device<IndexType> sorted_indexes;

        // binary radix tree
        std::unique_ptr<cuda_binary_radix_tree_t<IndexType, FloatType>> brt;
        IndexType num_bits_per_dim;
    };

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    void cuda_region_query(const vector_device<point3d_t<FloatType>>& positions, 
        const FloatType cutoff, const cuda_region_query_t<MortonCodeType, IndexType, FloatType>& region_query, 
        const lattice_t<FloatType>* lattice, const vector_device<atom_t<FloatType>>& atoms,
        vector_device<cuda_query_result_t<FloatType>>& query_results, vector_device<IndexType>& query_offsets,
        cudaStream_t stream);

}}
