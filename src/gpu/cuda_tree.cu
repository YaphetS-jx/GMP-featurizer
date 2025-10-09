#include "cuda_tree.hpp"
#include "resources.hpp"
#include "cuda_thrust_ops.hpp"
#include "gmp_float.hpp"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include "cuda_util.hpp"

namespace gmp { namespace tree {

    using namespace gmp::thrust_ops;

    // define constant memory for check_sphere_constant
    __constant__ cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t> check_sphere_constant;

    // cuda_check_sphere_t implementations
    template <typename MortonCodeType, typename FloatType, typename IndexType>
    __device__ 
    bool cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>::operator()(
        const array3d_t<FloatType>& lower_coords, const array3d_t<FloatType>& upper_coords, 
        const point3d_t<FloatType>& position, const array3d_t<IndexType>& cell_shift) const 
    {
        array3d_t<FloatType> difference = {
            gmp::util::get_difference(lower_coords[0] + cell_shift[0], upper_coords[0] + cell_shift[0], position.x), 
            gmp::util::get_difference(lower_coords[1] + cell_shift[1], upper_coords[1] + cell_shift[1], position.y),
            gmp::util::get_difference(lower_coords[2] + cell_shift[2], upper_coords[2] + cell_shift[2], position.z)
        };
        auto distance_squared = gmp::util::cuda_calculate_distance_squared(metric, difference);
        return distance_squared <= radius2;
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType>
    __device__ 
    void cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>::operator()(
        const array3d_t<FloatType> leaf_coords, const IndexType idx, 
        const point3d_t<FloatType>& position, const array3d_t<IndexType>& cell_shift, 
        IndexType* indexes, IndexType* num_indexes, const IndexType indexes_offset) const 
    {
        FloatType size_per_dim = 1.0f / (1 << (num_bits_per_dim - 1));
        
        const auto x_min_shift = leaf_coords[0] + cell_shift[0];
        const auto y_min_shift = leaf_coords[1] + cell_shift[1];
        const auto z_min_shift = leaf_coords[2] + cell_shift[2];
        const auto x_max_shift = x_min_shift + size_per_dim;
        const auto y_max_shift = y_min_shift + size_per_dim;
        const auto z_max_shift = z_min_shift + size_per_dim;
        array3d_t<FloatType> difference = {
            gmp::util::get_difference(x_min_shift, x_max_shift, position.x), 
            gmp::util::get_difference(y_min_shift, y_max_shift, position.y),
            gmp::util::get_difference(z_min_shift, z_max_shift, position.z)
        };

        auto distance_squared = gmp::util::cuda_calculate_distance_squared(metric, difference);
        if (distance_squared <= radius2) {
            if (indexes != nullptr) {
                indexes[indexes_offset + *num_indexes] = idx;
            }
            (*num_indexes)++;
        }
    }

    template class cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>;

    // binary radix tree implementations
    template <typename MortonCodeType, typename IndexType, typename FloatType>
    __global__
    void build_tree_kernel_internal(const MortonCodeType* morton_codes, const IndexType num_mc, const IndexType num_bits, internal_node_t<IndexType, FloatType>* internal_nodes)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_mc - 1) return;  // We have num_mc - 1 internal nodes

        IndexType first, last;
        morton_codes::determine_range<MortonCodeType, IndexType>(morton_codes, num_mc, tid, first, last, num_bits);
        IndexType delta_node = morton_codes::delta<MortonCodeType, IndexType>(morton_codes, num_mc, first, last, num_bits);
        IndexType split = morton_codes::find_split<MortonCodeType, IndexType>(morton_codes, num_mc, delta_node, first, last, num_bits);
        MortonCodeType lower_bound, upper_bound;
        morton_codes::find_lower_upper_bounds<MortonCodeType, IndexType>(morton_codes[split], delta_node, lower_bound, upper_bound, num_bits);

        // Determine left and right children
        // n is the number of leaf nodes (num_mc)
        IndexType n = num_mc;
        IndexType left = (split == first) ? split : split + n;
        IndexType right = (split + 1 == last) ? split + 1 : split + 1 + n;

        // Create internal node with float coordinates
        internal_node_t<IndexType, FloatType> node;
        
        // calculate internal node range 
        const IndexType num_bits_per_dim = num_bits / 3;
        MortonCodeType x_min, y_min, z_min;
        deinterleave_bits(lower_bound, num_bits_per_dim, x_min, y_min, z_min);
        MortonCodeType x_max, y_max, z_max;
        deinterleave_bits(upper_bound, num_bits_per_dim, x_max, y_max, z_max);
        
        // Convert morton code components to float coordinates
        FloatType size_per_dim = FloatType(1.0) / (1 << (num_bits_per_dim - 1));
        node.lower_bound_coords = array3d_t<FloatType>{
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim),
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim),
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim)
        };
        node.upper_bound_coords = array3d_t<FloatType>{
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_max, num_bits_per_dim) + size_per_dim,
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_max, num_bits_per_dim) + size_per_dim,
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_max, num_bits_per_dim) + size_per_dim
        };
        
        // Set left and right indices using the padding field
        node.set_indices(left, right);

        internal_nodes[tid] = node;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    __global__
    void build_tree_kernel_leaf(const MortonCodeType* morton_codes, const IndexType num_mc, const IndexType num_bits_per_dim, array3d_t<FloatType>* leaf_nodes)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_mc) return;  // We have num_mc leaf nodes

        // calculate leaf nodes
        MortonCodeType x_min, y_min, z_min;
        deinterleave_bits(morton_codes[tid], num_bits_per_dim, x_min, y_min, z_min);
        array3d_t<FloatType> leaf_node = {
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim),
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim),
            morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim)
        };
        leaf_nodes[tid] = leaf_node;
    }

    template <typename IndexType, typename FloatType>
    template <typename MortonCodeType>
    cuda_binary_radix_tree_t<IndexType, FloatType>::cuda_binary_radix_tree_t(
        const vector<MortonCodeType>& h_morton_codes, const IndexType num_bits, cudaStream_t stream)
        : num_leaf_nodes(h_morton_codes.size()), 
        internal_nodes(num_leaf_nodes - 1, stream), 
        leaf_nodes(num_leaf_nodes, stream)
    {
        assert(num_bits % 3 == 0);

        // Create temporary device memory for morton codes
        vector_device<MortonCodeType> d_morton_codes(num_leaf_nodes, stream);
        cudaMemcpyAsync(d_morton_codes.data(), h_morton_codes.data(), num_leaf_nodes * sizeof(MortonCodeType), cudaMemcpyHostToDevice, stream);

        dim3 block_size(256, 1, 1), grid_size(1, 1, 1);
        grid_size.x = (num_leaf_nodes - 1 + block_size.x - 1) / block_size.x;
        build_tree_kernel_internal<MortonCodeType, IndexType, FloatType><<<grid_size, block_size, 0, stream>>>(d_morton_codes.data(), num_leaf_nodes, num_bits, internal_nodes.data());
        grid_size.x = (num_leaf_nodes + block_size.x - 1) / block_size.x;
        build_tree_kernel_leaf<MortonCodeType, IndexType, FloatType><<<grid_size, block_size, 0, stream>>>(d_morton_codes.data(), num_leaf_nodes, num_bits/3, leaf_nodes.data());
    }

    template class cuda_binary_radix_tree_t<int32_t, gmp::gmp_float>;
    template cuda_binary_radix_tree_t<int32_t, gmp::gmp_float>::cuda_binary_radix_tree_t<uint32_t>(const vector<uint32_t>&, const int32_t, cudaStream_t);

    template <typename IndexType, typename FloatType>
    void cuda_binary_radix_tree_t<IndexType, FloatType>::get_internal_nodes(vector<inode_t>& h_internal_nodes) const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        h_internal_nodes.resize(num_leaf_nodes - 1);
        cudaMemcpyAsync(h_internal_nodes.data(), internal_nodes.data(), (num_leaf_nodes - 1) * sizeof(inode_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    template <typename IndexType, typename FloatType>
    void cuda_binary_radix_tree_t<IndexType, FloatType>::get_leaf_nodes(vector<array3d_t<FloatType>>& h_leaf_nodes) const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        h_leaf_nodes.resize(num_leaf_nodes);
        cudaMemcpyAsync(h_leaf_nodes.data(), leaf_nodes.data(), num_leaf_nodes * sizeof(array3d_t<FloatType>), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType>
    __device__
    void cuda_tree_traverse(const internal_node_t<IndexType, FloatType>* __restrict__ internal_nodes, 
        const array3d_t<FloatType>* __restrict__ leaf_nodes, const IndexType num_leaf_nodes, 
        const Checker check_method, const point3d_t<FloatType> position, 
        const array3d_t<IndexType> cell_shift, IndexType* __restrict__ indexes, 
        IndexType* num_indexes, const IndexType indexes_offset)
    {
        // Fixed stack for traversal
        IndexType stack_data[64];
        int stack_top = -1;
        
        IndexType result_index = 0;
        IndexType n = num_leaf_nodes;
        
        // Start with root (internal node index n)
        stack_data[++stack_top] = n;
        
        while (stack_top >= 0) {
            IndexType node_index = stack_data[stack_top--];
            
            if (node_index < n) {
                // Leaf node
                const auto& leaf_coords = leaf_nodes[node_index];
                
                // Check if leaf coords is within query bounds
                check_method(leaf_coords, node_index, position, cell_shift, indexes, num_indexes, indexes_offset);
            } else {
                // Internal node
                const internal_node_t<IndexType, FloatType>& node = internal_nodes[node_index - n];
                IndexType left = node.get_left();
                IndexType right = node.get_right();
                
                // Use float coordinates directly for checking
                if (check_method(node.lower_bound_coords, node.upper_bound_coords, position, cell_shift)) {
                    if (stack_top < 63) stack_data[++stack_top] = left;
                    if (stack_top < 63) stack_data[++stack_top] = right;
                }
            }
        }
        return;
    }

    template __device__
    void cuda_tree_traverse<cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t>
    (const internal_node_t<int32_t, gmp::gmp_float>* internal_nodes, const array3d_t<gmp::gmp_float>* leaf_nodes, const int32_t num_leaf_nodes, 
        const cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t> check_method, 
        const point3d_t<gmp::gmp_float> position, const array3d_t<int32_t> cell_shift, int32_t* indexes, int32_t* num_indexes, const int32_t indexes_offset);

    template __device__
    void cuda_tree_traverse<cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t>
    (const internal_node_t<int32_t, gmp::gmp_float>* internal_nodes, const array3d_t<gmp::gmp_float>* leaf_nodes, const int32_t num_leaf_nodes, 
        const cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t> check_method, 
        const point3d_t<gmp::gmp_float> position, const array3d_t<int32_t> cell_shift, int32_t* indexes, int32_t* num_indexes, const int32_t indexes_offset);



    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType, int MAX_STACK>
    __global__
    void cuda_tree_traverse_warp(const internal_node_t<IndexType, FloatType>* internal_nodes, const array3d_t<FloatType>* leaf_nodes, const IndexType num_leaf_nodes, 
        const Checker& check_method, const point3d_t<FloatType>* positions, const IndexType* query_target_indexes,
        const array3d_t<IndexType>* cell_shifts, const IndexType num_queries,
        IndexType* indexes, IndexType* num_indexes, const IndexType* num_indexes_offset)
    {
        const IndexType lane  = threadIdx.x & 31;
        // const IndexType warp  = threadIdx.x >> 5;
        // const IndexType warps_per_block = blockDim.x >> 5;
        // const IndexType global_warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
        const IndexType q0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) << 5;
        if (q0 >= num_queries) return; // warp level inactive, skip

        const IndexType tid = q0 + lane;
        IndexType packet_mask = __ballot_sync(0xffffffff, tid < num_queries);
        if (!packet_mask) return; // warp level inactive, skip

        // const auto& query_target_index = tid < num_queries ? query_target_indexes[tid] : 0;
        const auto& position = positions[tid < num_queries ? query_target_indexes[tid] : 0];
        const auto& cell_shift = tid < num_queries ? cell_shifts[tid] : array3d_t<IndexType>{0, 0, 0};

        extern __shared__ IndexType smem[];
        IndexType *stack_nodes = smem;                                  // [warps_per_block * MAX_STACK]
        IndexType *stack_masks = smem + (blockDim.x >> 5) * MAX_STACK;    // [warps_per_block * MAX_STACK]

        auto sidx = [&](IndexType sp)->IndexType { return (threadIdx.x >> 5) * MAX_STACK + sp; };
        IndexType stack_top = -1;
        if (lane == 0) {
            ++stack_top;
            stack_nodes[sidx(stack_top)] = num_leaf_nodes;
            stack_masks[sidx(stack_top)] = packet_mask;
        }
        // broadcast stack_top
        // assert((packet_mask & 1u));
        stack_top = __shfl_sync(0xffffffff, stack_top, 0);
        
        while (stack_top >= 0) {
            IndexType node_index, mask;
            if (lane == 0) {
                node_index = stack_nodes[sidx(stack_top)];
                mask = stack_masks[sidx(stack_top)];
                --stack_top;
            }
            // broadcast node_index and mask
            node_index = __shfl_sync(0xffffffff, node_index, 0);
            mask = __shfl_sync(0xffffffff, mask, 0);
            // broadcast stack_top
            stack_top = __shfl_sync(0xffffffff, stack_top, 0);
            bool lane_active = ((unsigned)mask & (1u << lane)) != 0;
            
            if (node_index < num_leaf_nodes) {
                array3d_t<FloatType> leaf_coords = lane == 0 ? leaf_nodes[node_index] : array3d_t<FloatType>{0, 0, 0};
                leaf_coords[0] = __shfl_sync(0xffffffff, leaf_coords[0], 0);
                leaf_coords[1] = __shfl_sync(0xffffffff, leaf_coords[1], 0);
                leaf_coords[2] = __shfl_sync(0xffffffff, leaf_coords[2], 0);
                if (!lane_active) continue; 
                check_method(leaf_coords, node_index, position, cell_shift, indexes, 
                    (tid < num_queries) ? num_indexes+tid : nullptr, 
                    tid < num_queries ? (num_indexes_offset ? (tid > 0 ? num_indexes_offset[tid - 1] : 0) : 0) : 0);
            } else {
                // Access internal node using global memory
                const internal_node_t<IndexType, FloatType>& node = internal_nodes[node_index - num_leaf_nodes];
                
                // Use float coordinates directly for checking
                bool hit = lane_active && check_method(node.lower_bound_coords, node.upper_bound_coords, position, cell_shift);
                IndexType parent_mask  = __ballot_sync(0xffffffff, hit);

                if (lane == 0 && parent_mask) {
                    IndexType left = node.get_left();
                    IndexType right = node.get_right();

                    ++stack_top;
                    stack_nodes[sidx(stack_top)] = left; stack_masks[sidx(stack_top)] = parent_mask;
                    ++stack_top;
                    stack_nodes[sidx(stack_top)] = right; stack_masks[sidx(stack_top)] = parent_mask;
                }
                // broadcast stack_top
                stack_top = __shfl_sync(0xffffffff, stack_top, 0);
            }
        }
    }

    template __global__
    void cuda_tree_traverse_warp<cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t, 24>
    (const internal_node_t<int32_t, gmp::gmp_float>* internal_nodes, const array3d_t<gmp::gmp_float>* leaf_nodes, const int32_t num_leaf_nodes, 
        const cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t>& check_method, 
        const point3d_t<gmp::gmp_float>* positions, const int32_t* query_target_indexes, const array3d_t<int32_t>* cell_shifts, const int32_t num_queries,
        int32_t* indexes, int32_t* num_indexes, const int32_t* num_indexes_offset);

    template __global__
    void cuda_tree_traverse_warp<cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t, 24>
    (const internal_node_t<int32_t, gmp::gmp_float>* internal_nodes, const array3d_t<gmp::gmp_float>* leaf_nodes, const int32_t num_leaf_nodes, 
        const cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>& check_method, 
        const point3d_t<gmp::gmp_float>* positions, const int32_t* query_target_indexes, const array3d_t<int32_t>* cell_shifts, const int32_t num_queries,
        int32_t* indexes, int32_t* num_indexes, const int32_t* num_indexes_offset);


    template <typename MortonCodeType, typename FloatType, typename IndexType, int MAX_STACK>
    __global__
    void cuda_tree_traverse_warp2(const internal_node_t<IndexType, FloatType>* internal_nodes, const array3d_t<FloatType>* leaf_nodes, const IndexType num_leaf_nodes, 
        const point3d_t<FloatType>* positions, const IndexType* query_target_indexes,
        const array3d_t<IndexType>* cell_shifts, const IndexType num_queries,
        IndexType* indexes, IndexType* num_indexes, const IndexType* num_indexes_offset)
    {
        const IndexType lane  = threadIdx.x & 31;
        // const IndexType warp  = threadIdx.x >> 5;
        // const IndexType warps_per_block = blockDim.x >> 5;
        // const IndexType global_warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
        const IndexType q0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) << 5;
        if (q0 >= num_queries) return; // warp level inactive, skip

        const IndexType tid = q0 + lane;
        IndexType packet_mask = __ballot_sync(0xffffffff, tid < num_queries);
        if (!packet_mask) return; // warp level inactive, skip

        const auto& position = positions[tid < num_queries ? query_target_indexes[tid] : 0];
        const auto& cell_shift = tid < num_queries ? cell_shifts[tid] : array3d_t<IndexType>{0, 0, 0};
        // IndexType dummy_num_index = 0;
        // IndexType* num_index = tid < num_queries ? num_indexes+tid : nullptr;
        // const IndexType indexes_offset = tid < num_queries ? (num_indexes_offset ? (tid > 0 ? num_indexes_offset[tid - 1] : 0) : 0) : 0;

        extern __shared__ IndexType smem[];
        IndexType *stack_nodes = smem;                                  // [warps_per_block * MAX_STACK]
        IndexType *stack_masks = smem + (blockDim.x >> 5) * MAX_STACK;    // [warps_per_block * MAX_STACK]

        auto sidx = [&](IndexType sp)->IndexType { return (threadIdx.x >> 5) * MAX_STACK + sp; };
        IndexType stack_top = -1;
        if (lane == 0) {
            ++stack_top;
            stack_nodes[sidx(stack_top)] = num_leaf_nodes;
            stack_masks[sidx(stack_top)] = packet_mask;
        }
        // broadcast stack_top
        // assert((packet_mask & 1u));
        stack_top = __shfl_sync(0xffffffff, stack_top, 0);
        
        while (stack_top >= 0) {
            IndexType node_index, mask;
            if (lane == 0) {
                node_index = stack_nodes[sidx(stack_top)];
                mask = stack_masks[sidx(stack_top)];
                --stack_top;
            }
            // broadcast node_index and mask
            node_index = __shfl_sync(0xffffffff, node_index, 0);
            mask = __shfl_sync(0xffffffff, mask, 0);
            // broadcast stack_top
            stack_top = __shfl_sync(0xffffffff, stack_top, 0);
            bool lane_active = ((unsigned)mask & (1u << lane)) != 0;
            
            if (node_index < num_leaf_nodes) {
                array3d_t<FloatType> leaf_coords = lane == 0 ? leaf_nodes[node_index] : array3d_t<FloatType>{0, 0, 0};
                leaf_coords[0] = __shfl_sync(0xffffffff, leaf_coords[0], 0);
                leaf_coords[1] = __shfl_sync(0xffffffff, leaf_coords[1], 0);
                leaf_coords[2] = __shfl_sync(0xffffffff, leaf_coords[2], 0);
                if (!lane_active) continue; 
                check_sphere_constant(leaf_coords, node_index, position, cell_shift, indexes, 
                    (tid < num_queries) ? num_indexes+tid : nullptr, 
                    tid < num_queries ? (num_indexes_offset ? (tid > 0 ? num_indexes_offset[tid - 1] : 0) : 0) : 0);
            } else {
                // Access internal node using global memory
                const internal_node_t<IndexType, FloatType>& node = internal_nodes[node_index - num_leaf_nodes];

                bool hit = lane_active && check_sphere_constant(node.lower_bound_coords, node.upper_bound_coords, position, cell_shift);
                IndexType parent_mask  = __ballot_sync(0xffffffff, hit);

                if (lane == 0 && parent_mask) {
                    IndexType left = node.get_left();
                    IndexType right = node.get_right();

                    ++stack_top;
                    stack_nodes[sidx(stack_top)] = left; stack_masks[sidx(stack_top)] = parent_mask;
                    ++stack_top;
                    stack_nodes[sidx(stack_top)] = right; stack_masks[sidx(stack_top)] = parent_mask;
                }
                // broadcast stack_top
                stack_top = __shfl_sync(0xffffffff, stack_top, 0);
            }
        }
    }

    template __global__
    void cuda_tree_traverse_warp2<uint32_t, gmp::gmp_float, int32_t, 24>
    (const internal_node_t<int32_t, gmp::gmp_float>* internal_nodes, const array3d_t<gmp::gmp_float>* leaf_nodes, const int32_t num_leaf_nodes, 
        const point3d_t<gmp::gmp_float>* positions, const int32_t* query_target_indexes, 
        const array3d_t<int32_t>* cell_shifts, const int32_t num_queries,
        int32_t* indexes, int32_t* num_indexes, const int32_t* num_indexes_offset);
        
}} // namespace gmp::tree 