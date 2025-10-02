#include "cuda_tree.hpp"
#include "resources.hpp"
#include "cuda_thrust_ops.hpp"
#include "gmp_float.hpp"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include "cuda_util.hpp"

namespace gmp { namespace tree {

    using namespace gmp::thrust_ops;

    // cuda_check_sphere_t implementations
    template <typename MortonCodeType, typename FloatType, typename IndexType>
    __device__ 
    bool cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>::operator()(
        const MortonCodeType lower_bound, const MortonCodeType upper_bound, 
        const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift) const 
    {
        MortonCodeType x_min, y_min, z_min;
        deinterleave_bits(lower_bound, num_bits_per_dim, x_min, y_min, z_min);
        MortonCodeType x_max, y_max, z_max;
        deinterleave_bits(upper_bound, num_bits_per_dim, x_max, y_max, z_max);

        auto x_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim);
        auto y_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim);
        auto z_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim);

        FloatType size_per_dim = 1.0f / (1 << (num_bits_per_dim - 1));
        auto x_max_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_max, num_bits_per_dim) + size_per_dim;
        auto y_max_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_max, num_bits_per_dim) + size_per_dim;
        auto z_max_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_max, num_bits_per_dim) + size_per_dim;

        auto get_difference = [](FloatType min, FloatType max, FloatType point) {
            return (min <= point && point <= max) ? 0 : (point < min) ? min - point : point - max;
        };

        array3d_t<FloatType> difference = {
            get_difference(x_min_f + cell_shift[0], x_max_f + cell_shift[0], position.x), 
            get_difference(y_min_f + cell_shift[1], y_max_f + cell_shift[1], position.y),
            get_difference(z_min_f + cell_shift[2], z_max_f + cell_shift[2], position.z)
        };
        auto distance_squared = gmp::util::cuda_calculate_distance_squared(metric, difference);
        return distance_squared <= radius2;
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType>
    __device__ 
    void cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>::operator()(
        const MortonCodeType morton_code, const IndexType idx, 
        const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift, 
        IndexType* indexes, IndexType& num_indexes, const IndexType indexes_offset) const 
    {
        MortonCodeType x_min, y_min, z_min;
        deinterleave_bits(morton_code, num_bits_per_dim, x_min, y_min, z_min);
        
        FloatType size_per_dim = 1.0f / (1 << (num_bits_per_dim - 1));
        auto x_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim);
        auto y_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim);
        auto z_min_f = morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim);
        auto x_max_f = x_min_f + size_per_dim;
        auto y_max_f = y_min_f + size_per_dim;
        auto z_max_f = z_min_f + size_per_dim;

        auto get_difference = [](FloatType min, FloatType max, FloatType point) {
            return (min <= point && point <= max) ? 0 : (point < min) ? min - point : point - max;
        };

        array3d_t<FloatType> difference = {
            get_difference(x_min_f + cell_shift[0], x_max_f + cell_shift[0], position.x), 
            get_difference(y_min_f + cell_shift[1], y_max_f + cell_shift[1], position.y),
            get_difference(z_min_f + cell_shift[2], z_max_f + cell_shift[2], position.z)
        };

        auto distance_squared = gmp::util::cuda_calculate_distance_squared(metric, difference);
        if (distance_squared <= radius2) {
            if (indexes != nullptr) {
                indexes[indexes_offset + num_indexes] = idx;    
            }
            num_indexes++;
        }
    }

    template class cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>;

    // binary radix tree implementations
    template <typename MortonCodeType, typename IndexType>
    __global__
    void build_tree_kernel(const MortonCodeType* morton_codes, const IndexType num_mc, const IndexType num_bits, internal_node_t<MortonCodeType, IndexType>* internal_nodes)
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

        internal_nodes[tid] = internal_node_t<MortonCodeType, IndexType>{left, right, lower_bound, upper_bound};
    }

    template <typename MortonCodeType, typename IndexType>
    cuda_binary_radix_tree_t<MortonCodeType, IndexType>::cuda_binary_radix_tree_t(
        const vector<MortonCodeType>& h_morton_codes, const IndexType num_bits, cudaStream_t stream)
        : num_leaf_nodes(h_morton_codes.size()), 
        internal_nodes(num_leaf_nodes - 1, stream), 
        leaf_nodes(num_leaf_nodes, stream)
    {
        assert(num_bits % 3 == 0);

        // copy leaf nodes
        cudaMemcpyAsync(leaf_nodes.data(), h_morton_codes.data(), num_leaf_nodes * sizeof(MortonCodeType), cudaMemcpyHostToDevice, stream);

        dim3 block_size(256, 1, 1), grid_size(1, 1, 1);
        grid_size.x = (num_leaf_nodes - 1 + block_size.x - 1) / block_size.x;
        build_tree_kernel<MortonCodeType, IndexType><<<grid_size, block_size, 0, stream>>>(leaf_nodes.data(), num_leaf_nodes, num_bits, internal_nodes.data());

        bind_texture_memory(internal_nodes.data(), (num_leaf_nodes - 1) * sizeof(inode_t), 32, internal_nodes_tex);
        bind_texture_memory(leaf_nodes.data(), num_leaf_nodes * sizeof(MortonCodeType), 32, leaf_nodes_tex);
    }

    template <typename MortonCodeType, typename IndexType>
    cuda_binary_radix_tree_t<MortonCodeType, IndexType>::~cuda_binary_radix_tree_t()
    {
        unbind_texture_memory(internal_nodes_tex);
        unbind_texture_memory(leaf_nodes_tex);
    }

    template class cuda_binary_radix_tree_t<uint32_t, int32_t>;

    template <typename MortonCodeType, typename IndexType>
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType>::get_internal_nodes(vector<inode_t>& h_internal_nodes) const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        h_internal_nodes.resize(num_leaf_nodes - 1);
        cudaMemcpyAsync(h_internal_nodes.data(), internal_nodes.data(), (num_leaf_nodes - 1) * sizeof(inode_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    template <typename MortonCodeType, typename IndexType>
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType>::get_leaf_nodes(vector<MortonCodeType>& h_leaf_nodes) const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        h_leaf_nodes.resize(num_leaf_nodes);
        cudaMemcpyAsync(h_leaf_nodes.data(), leaf_nodes.data(), num_leaf_nodes * sizeof(MortonCodeType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType>
    __device__
    void cuda_tree_traverse(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, const IndexType num_leaf_nodes, 
        const Checker check_method, const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift, IndexType* indexes, IndexType& num_indexes, const IndexType indexes_offset)
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
                MortonCodeType morton_code = tex1Dfetch<MortonCodeType>(leaf_nodes_tex, node_index);
                
                // Check if morton code is within query bounds
                check_method(morton_code, node_index, position, cell_shift, indexes, num_indexes, indexes_offset);
            } else {
                // Internal node
                IndexType left = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - n) * 4);
                IndexType right = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - n) * 4 + 1);
                MortonCodeType lower_bound = tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - n) * 4 + 2);
                MortonCodeType upper_bound = tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - n) * 4 + 3);
                
                if (check_method(lower_bound, upper_bound, position, cell_shift)) {
                    if (stack_top < 63) stack_data[++stack_top] = left;
                    if (stack_top < 63) stack_data[++stack_top] = right;
                }
            }
        }
        return;
    }

    template __device__
    void cuda_tree_traverse<cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t>
    (const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, const int32_t num_leaf_nodes, 
        const cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t> check_method, 
        const point3d_t<gmp::gmp_float> position, const array3d_t<int32_t> cell_shift, int32_t* indexes, int32_t& num_indexes, const int32_t indexes_offset);

    template __device__
    void cuda_tree_traverse<cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t>
    (const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, const int32_t num_leaf_nodes, 
        const cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t> check_method, 
        const point3d_t<gmp::gmp_float> position, const array3d_t<int32_t> cell_shift, int32_t* indexes, int32_t& num_indexes, const int32_t indexes_offset);

    void bind_texture_memory(void* data_ptr, uint32_t size, int bits_per_channel, cudaTextureObject_t& tex)
    {
        // Create texture descriptor for internal nodes
        cudaResourceDesc resDesc_internal = {};
        resDesc_internal.resType = cudaResourceTypeLinear;
        resDesc_internal.res.linear.devPtr = data_ptr;
        resDesc_internal.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        resDesc_internal.res.linear.desc.x = bits_per_channel; // bits per channel
        resDesc_internal.res.linear.sizeInBytes = size;
        
        cudaTextureDesc texDesc = {};
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        
        cudaCreateTextureObject(&tex, &resDesc_internal, &texDesc, nullptr);
    }

    void unbind_texture_memory(cudaTextureObject_t tex)
    {
        cudaDestroyTextureObject(tex);
    }


    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType, int MAX_STACK>
    __global__
    void cuda_tree_traverse_warp(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, const IndexType num_leaf_nodes, 
        const Checker& check_method, const point3d_t<FloatType>* positions, const IndexType* query_target_indexes,
        const array3d_t<IndexType>* cell_shifts, const IndexType num_queries,
        IndexType* indexes, IndexType* num_indexes, const IndexType* num_indexes_offset)
    {
        const IndexType lane  = threadIdx.x & 31;
        const IndexType warp  = threadIdx.x >> 5;
        const IndexType warps_per_block = blockDim.x >> 5;
        const IndexType global_warp = blockIdx.x * warps_per_block + warp;
        const IndexType q0 = global_warp * 32;
        if (q0 >= num_queries) return; // warp level inactive, skip

        const IndexType tid = q0 + lane;
        IndexType packet_mask = __ballot_sync(0xffffffff, tid < num_queries);
        if (!packet_mask) return; // warp level inactive, skip

        const auto& query_target_index = tid < num_queries ? query_target_indexes[tid] : 0;
        const auto& position = positions[query_target_index];
        const auto& cell_shift = tid < num_queries ? cell_shifts[tid] : array3d_t<IndexType>{0, 0, 0};
        IndexType dummy_num_index = 0;
        IndexType& num_index = (tid < num_queries) ? num_indexes[tid] : dummy_num_index;
        const IndexType indexes_offset = tid < num_queries ? (num_indexes_offset ? (tid > 0 ? num_indexes_offset[tid - 1] : 0) : 0) : 0;

        extern __shared__ IndexType smem[];
        IndexType *stack_nodes = smem;                                  // [warps_per_block * MAX_STACK]
        IndexType *stack_masks = smem + warps_per_block * MAX_STACK;    // [warps_per_block * MAX_STACK]

        auto sidx = [&](IndexType sp)->IndexType { return warp * MAX_STACK + sp; };
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
                MortonCodeType morton_code = lane == 0 ? tex1Dfetch<MortonCodeType>(leaf_nodes_tex, node_index) : 0;
                morton_code = __shfl_sync(0xffffffff, morton_code, 0);
                if (!lane_active) continue; 
                check_method(morton_code, node_index, position, cell_shift, indexes, num_index, indexes_offset);
            } else {
                MortonCodeType lower_bound = lane == 0 ? tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - num_leaf_nodes) * 4 + 2) : 0;
                MortonCodeType upper_bound = lane == 0 ? tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - num_leaf_nodes) * 4 + 3) : 0;
                lower_bound = __shfl_sync(0xffffffff, lower_bound, 0);
                upper_bound = __shfl_sync(0xffffffff, upper_bound, 0);

                bool hit = lane_active && check_method(lower_bound, upper_bound, position, cell_shift);
                IndexType parent_mask  = __ballot_sync(0xffffffff, hit);

                if (lane == 0 && parent_mask) {
                    IndexType left = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - num_leaf_nodes) * 4);
                    IndexType right = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - num_leaf_nodes) * 4 + 1);

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
    (const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, const int32_t num_leaf_nodes, 
        const cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t>& check_method, 
        const point3d_t<gmp::gmp_float>* positions, const int32_t* query_target_indexes, const array3d_t<int32_t>* cell_shifts, const int32_t num_queries,
        int32_t* indexes, int32_t* num_indexes, const int32_t* num_indexes_offset);

    template __global__
    void cuda_tree_traverse_warp<cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t, 24>
    (const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, const int32_t num_leaf_nodes, 
        const cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>& check_method, 
        const point3d_t<gmp::gmp_float>* positions, const int32_t* query_target_indexes, const array3d_t<int32_t>* cell_shifts, const int32_t num_queries,
        int32_t* indexes, int32_t* num_indexes, const int32_t* num_indexes_offset);
        
}} // namespace gmp::tree 