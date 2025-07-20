#include "cuda_region_query.hpp"
#include <algorithm>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace gmp { namespace region_query {



    // cuda_check_sphere_t implementations
    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    cuda_check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::cuda_check_sphere_t(
        const IndexType num_bits_per_dim, const array3d_bool periodicity, const lattice_t<FloatType>* lattice)
        : num_bits_per_dim(num_bits_per_dim), periodicity(periodicity), lattice(lattice) 
    {}

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    void cuda_check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::update_point_radius(
        point3d_t<FloatType> position_in, FloatType radius) {
        // get fractional radius 
        radius2 = radius * radius;
        frac_radius = radius / lattice->get_cell_lengths();
        position = position_in;

        // Initialize cell shifts
        cell_shift_start[0] = std::floor(position.x - frac_radius[0]);
        cell_shift_start[0] = std::floor(position.x - frac_radius[0]);
        cell_shift_start[1] = std::floor(position.y - frac_radius[1]);
        cell_shift_end[1] = std::floor(position.y + frac_radius[1]);
        cell_shift_start[2] = std::floor(position.z - frac_radius[2]);
        cell_shift_end[2] = std::floor(position.z + frac_radius[2]);
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    array3d_t<IndexType> cuda_check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::get_cell_shift_start() const {
        return cell_shift_start;
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType, typename VecType>
    array3d_t<IndexType> cuda_check_sphere_t<MortonCodeType, FloatType, IndexType, VecType>::get_cell_shift_end() const {
        return cell_shift_end;
    }

    // cuda_region_query_t implementations
    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    cuda_region_query_t<MortonCodeType, IndexType, FloatType, VecType>::cuda_region_query_t(
        const unit_cell_t<FloatType>* unit_cell, const uint8_t num_bits_per_dim, cudaStream_t stream) 
        : unique_morton_codes(0, stream),
          offsets(0, stream),
          sorted_indexes(0, stream),
          atom_positions(0, stream),
          num_bits_per_dim(num_bits_per_dim), 
          unit_cell_ptr(unit_cell) {
        
        // Get stream if not provided
        if (stream == 0) {
            stream = gmp::resources::gmp_resource::instance().get_stream();
        }
        
        // get morton codes
        get_morton_codes(unit_cell->get_atoms(), num_bits_per_dim, stream);
        
        // build tree
        brt = std::make_unique<cuda_binary_radix_tree_t<MortonCodeType, IndexType>>(unique_morton_codes, num_bits_per_dim * 3, stream);
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    void cuda_region_query_t<MortonCodeType, IndexType, FloatType, VecType>::get_morton_codes(
        const vector<atom_t<FloatType>>& atoms, const uint8_t num_bits_per_dim, cudaStream_t stream) {
        
        vector<MortonCodeType> morton_codes;
        auto natom = atoms.size();
        morton_codes.reserve(natom);
        
        // Generate morton codes on host
        for (const auto& atom : atoms) {
            MortonCodeType mc_x = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.x(), num_bits_per_dim);
            MortonCodeType mc_y = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.y(), num_bits_per_dim);
            MortonCodeType mc_z = coordinate_to_morton_code<FloatType, MortonCodeType, IndexType>(atom.z(), num_bits_per_dim);
            morton_codes.push_back(interleave_bits(mc_x, mc_y, mc_z, num_bits_per_dim));
        }

        // index mapping from morton codes to atoms
        vector<IndexType> sorted_indexes_host = util::sort_indexes<MortonCodeType, IndexType, vector>(morton_codes);
        std::sort(morton_codes.begin(), morton_codes.end());

        // compact
        vector<MortonCodeType> unique_morton_codes_host;
        unique_morton_codes_host.reserve(natom);
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
                unique_morton_codes_host.push_back(morton_codes[i]);
            }
        }

        // get offsets
        vector<IndexType> offsets_host;
        offsets_host.reserve(natom + 1);
        for (auto i = 0; i < natom; i++) {
            if (same[i]) {
                offsets_host.push_back(indexing[i]);
            }
        }
        offsets_host.push_back(natom);

        // Copy to device
        unique_morton_codes.resize(unique_morton_codes_host.size(), stream);
        offsets.resize(offsets_host.size(), stream);
        sorted_indexes.resize(sorted_indexes_host.size(), stream);
        
        cudaMemcpyAsync(unique_morton_codes.data(), unique_morton_codes_host.data(), 
            unique_morton_codes_host.size() * sizeof(MortonCodeType), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(offsets.data(), offsets_host.data(), 
            offsets_host.size() * sizeof(IndexType), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(sorted_indexes.data(), sorted_indexes_host.data(), 
            sorted_indexes_host.size() * sizeof(IndexType), cudaMemcpyHostToDevice, stream);

        // Copy atom positions to device
        vector<point3d_t<FloatType>> atom_positions_host;
        atom_positions_host.reserve(natom);
        for (const auto& atom : atoms) {
            atom_positions_host.push_back(atom.pos());
        }
        
        atom_positions.resize(atom_positions_host.size(), stream);
        cudaMemcpyAsync(atom_positions.data(), atom_positions_host.data(), 
            atom_positions_host.size() * sizeof(point3d_t<FloatType>), cudaMemcpyHostToDevice, stream);
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    const vector_device<MortonCodeType>& cuda_region_query_t<MortonCodeType, IndexType, FloatType, VecType>::get_unique_morton_codes() const {
        return unique_morton_codes;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    const vector_device<IndexType>& cuda_region_query_t<MortonCodeType, IndexType, FloatType, VecType>::get_offsets() const {
        return offsets;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    const vector_device<IndexType>& cuda_region_query_t<MortonCodeType, IndexType, FloatType, VecType>::get_sorted_indexes() const {
        return sorted_indexes;
    }

    // CUDA kernel for distance calculation and filtering for candidate indices with shifts
    // Each thread processes one candidate atom index and applies the corresponding shift
    // candidate_indices: indices into sorted_indexes (i.e., leaf indices from tree traversal)
    // candidate_shifts: 3*len array, each triple is the shift for the corresponding candidate
    // atom_positions: fractional coordinates
    // shift is in fractional units
    //
    template <typename FloatType, typename IndexType>
    __global__ void calculate_distances_candidates_shifts_kernel(
        const point3d_t<FloatType>* atom_positions,
        const IndexType* sorted_indexes,
        const IndexType* candidate_indices,
        const int32_t* candidate_shifts,
        const point3d_t<FloatType> query_position,
        const FloatType cutoff_squared,
        IndexType* result_indices,
        FloatType* result_distances,
        IndexType* result_count,
        const IndexType num_candidates) {
        IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_candidates) return;

        IndexType sorted_idx = candidate_indices[tid];
        IndexType atom_index = sorted_indexes[sorted_idx];
        point3d_t<FloatType> atom_pos = atom_positions[atom_index];

        // Apply shift (in fractional coordinates)
        int32_t sx = candidate_shifts[tid * 3 + 0];
        int32_t sy = candidate_shifts[tid * 3 + 1];
        int32_t sz = candidate_shifts[tid * 3 + 2];
        FloatType dx = atom_pos.x + sx - query_position.x;
        FloatType dy = atom_pos.y + sy - query_position.y;
        FloatType dz = atom_pos.z + sz - query_position.z;
        FloatType distance2 = dx*dx + dy*dy + dz*dz;

        if (distance2 < cutoff_squared) {
            IndexType idx = atomicAdd(result_count, 1);
            result_indices[idx] = atom_index;
            result_distances[idx] = distance2;
        }
    }



    template <typename MortonCodeType, typename IndexType, typename FloatType, typename VecType>
    typename cuda_region_query_t<MortonCodeType, IndexType, FloatType, VecType>::result_t 
    cuda_region_query_t<MortonCodeType, IndexType, FloatType, VecType>::query(
        const point3d_t<FloatType>& position, const FloatType cutoff, 
        const unit_cell_t<FloatType>* unit_cell, cudaStream_t stream) {
        // Get stream if not provided
        if (stream == 0) {
            stream = gmp::resources::gmp_resource::instance().get_stream();
        }
        auto cutoff_squared = cutoff * cutoff;
        
        // For now, use a simple approach: get all regions from the tree
        // and check each one manually, similar to CPU but without the complex comparison operator
        std::vector<IndexType> all_candidate_indices_host;
        std::vector<int32_t> all_candidate_shifts_host;
        
        // Get cell shifts based on periodicity and cutoff
        auto frac_radius = cutoff / unit_cell->get_lattice()->get_cell_lengths();
        array3d_t<int32_t> cell_shift_start, cell_shift_end;
        cell_shift_start[0] = std::floor(position.x - frac_radius[0]);
        cell_shift_end[0] = std::floor(position.x + frac_radius[0]);
        cell_shift_start[1] = std::floor(position.y - frac_radius[1]);
        cell_shift_end[1] = std::floor(position.y + frac_radius[1]);
        cell_shift_start[2] = std::floor(position.z - frac_radius[2]);
        cell_shift_end[2] = std::floor(position.z + frac_radius[2]);
        
        // Copy unique_morton_codes and offsets to host for processing
        std::vector<MortonCodeType> unique_morton_codes_host(unique_morton_codes.size());
        std::vector<IndexType> offsets_host(offsets.size());
        cudaMemcpy(unique_morton_codes_host.data(), unique_morton_codes.data(), unique_morton_codes.size() * sizeof(MortonCodeType), cudaMemcpyDeviceToHost);
        cudaMemcpy(offsets_host.data(), offsets.data(), offsets.size() * sizeof(IndexType), cudaMemcpyDeviceToHost);
        
        // For each region in the tree, check if it could contain atoms within the cutoff
        for (IndexType region_index = 0; region_index < unique_morton_codes_host.size(); ++region_index) {
            // Get the Morton code for this region
            MortonCodeType region_mc = unique_morton_codes_host[region_index];
            
            // Deinterleave to get coordinate bounds
            MortonCodeType mc_x, mc_y, mc_z;
            deinterleave_bits(region_mc, num_bits_per_dim, mc_x, mc_y, mc_z);
            
            auto x_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(mc_x, num_bits_per_dim);
            auto y_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(mc_y, num_bits_per_dim);
            auto z_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(mc_z, num_bits_per_dim);
            
            FloatType size_per_dim = 1.0 / (1 << (num_bits_per_dim - 1));
            auto x_max_f = x_f + size_per_dim;
            auto y_max_f = y_f + size_per_dim;
            auto z_max_f = z_f + size_per_dim;
            
            // Check all possible shifts for this region
            for (int sz = cell_shift_start[2]; sz <= cell_shift_end[2]; ++sz) {
                for (int sy = cell_shift_start[1]; sy <= cell_shift_end[1]; ++sy) {
                    for (int sx = cell_shift_start[0]; sx <= cell_shift_end[0]; ++sx) {
                        // Check if this shifted region could intersect the sphere
                        auto x_min_shift = x_f + sx;
                        auto y_min_shift = y_f + sy;
                        auto z_min_shift = z_f + sz;
                        auto x_max_shift = x_max_f + sx;
                        auto y_max_shift = y_max_f + sy;
                        auto z_max_shift = z_max_f + sz;
                        
                        // Simple bounding box check: if the region is completely outside the sphere, skip it
                        if (x_max_shift < position.x - cutoff || x_min_shift > position.x + cutoff ||
                            y_max_shift < position.y - cutoff || y_min_shift > position.y + cutoff ||
                            z_max_shift < position.z - cutoff || z_min_shift > position.z + cutoff) {
                            continue;
                        }
                        
                        // This region and shift could contain atoms within the sphere, add all atoms in this region
                        IndexType start_idx = offsets_host[region_index];
                        IndexType end_idx = offsets_host[region_index + 1];
                        for (IndexType idx = start_idx; idx < end_idx; ++idx) {
                            all_candidate_indices_host.push_back(idx);
                            all_candidate_shifts_host.push_back(sx);
                            all_candidate_shifts_host.push_back(sy);
                            all_candidate_shifts_host.push_back(sz);
                        }
                    }
                }
            }
        }
        
        IndexType num_candidate_atoms = all_candidate_indices_host.size();
        if (num_candidate_atoms == 0) {
            return result_t(0, stream);
        }
        
        // Copy candidate indices and shifts to device
        vector_device<IndexType> candidate_indices(num_candidate_atoms, stream);
        vector_device<int32_t> candidate_shifts(num_candidate_atoms * 3, stream);
        cudaMemcpyAsync(candidate_indices.data(), all_candidate_indices_host.data(), num_candidate_atoms * sizeof(IndexType), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(candidate_shifts.data(), all_candidate_shifts_host.data(), num_candidate_atoms * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        
        // Allocate temporary arrays for results
        vector_device<IndexType> result_indices(num_candidate_atoms, stream);
        vector_device<FloatType> result_distances(num_candidate_atoms, stream);
        vector_device<IndexType> result_count(1, stream);
        
        // Initialize result count to 0
        thrust::fill(thrust::cuda::par.on(stream), result_count.begin(), result_count.end(), 0);
        
        // Launch kernel to calculate distances and filter
        dim3 block_size(256, 1, 1);
        dim3 grid_size((num_candidate_atoms + block_size.x - 1) / block_size.x, 1, 1);
        
        calculate_distances_candidates_shifts_kernel<<<grid_size, block_size, 0, stream>>>(
            atom_positions.data(),
            sorted_indexes.data(),
            candidate_indices.data(),
            candidate_shifts.data(),
            position,
            cutoff_squared,
            result_indices.data(),
            result_distances.data(),
            result_count.data(),
            num_candidate_atoms
        );
        
        // Get the actual number of results
        IndexType actual_count;
        cudaMemcpyAsync(&actual_count, result_count.data(), sizeof(IndexType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        if (actual_count == 0) {
            return result_t(0, stream);
        }
        
        // Create result with actual count
        result_t result(actual_count, stream);
        
        // Copy results to the final result vector
        cudaMemcpyAsync(result.data(), result_indices.data(), actual_count * sizeof(IndexType), cudaMemcpyDeviceToDevice, stream);
        
        // Sort the results by index to match CPU output
        thrust::sort(thrust::cuda::par.on(stream), result.begin(), result.end());
        
        return result;
    }
    


}} // namespace gmp::region_query 