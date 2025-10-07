#include "cuda_tree.hpp"
#include "resources.hpp"
#include "cuda_thrust_ops.hpp"
#include "gmp_float.hpp"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include "cuda_util.hpp"
#include <type_traits>

namespace gmp { namespace tree {

    using namespace gmp::thrust_ops;

    // cuda_check_sphere_t implementations
    template <typename MortonCodeType, typename FloatType, typename IndexType>
    __device__ 
    bool cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>::operator()(
        const array3d_t<FloatType>& min_bounds, const array3d_t<FloatType>& max_bounds,
        const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift) const
    {
        auto get_difference = [](FloatType min, FloatType max, FloatType point) {
            return (min <= point && point <= max) ? FloatType(0)
                : (point < min) ? min - point : point - max;
        };

        array3d_t<FloatType> difference = {
            get_difference(min_bounds[0] + cell_shift[0], max_bounds[0] + cell_shift[0], position.x),
            get_difference(min_bounds[1] + cell_shift[1], max_bounds[1] + cell_shift[1], position.y),
            get_difference(min_bounds[2] + cell_shift[2], max_bounds[2] + cell_shift[2], position.z)
        };
        auto distance_squared = gmp::util::cuda_calculate_distance_squared(metric, difference);
        return distance_squared <= radius2;
    }

    template <typename MortonCodeType, typename FloatType, typename IndexType>
    __device__ 
    void cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>::operator()(
        const array3d_t<FloatType>& min_bounds, const array3d_t<FloatType>& max_bounds,
        const point3d_t<FloatType> position, const array3d_t<IndexType> cell_shift,
        IndexType* indexes, IndexType& num_indexes, const IndexType indexes_offset) const
    {
        auto get_difference = [](FloatType min, FloatType max, FloatType point) {
            return (min <= point && point <= max) ? FloatType(0)
                : (point < min) ? min - point : point - max;
        };

        array3d_t<FloatType> difference = {
            get_difference(min_bounds[0] + cell_shift[0], max_bounds[0] + cell_shift[0], position.x),
            get_difference(min_bounds[1] + cell_shift[1], max_bounds[1] + cell_shift[1], position.y),
            get_difference(min_bounds[2] + cell_shift[2], max_bounds[2] + cell_shift[2], position.z)
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
    template <typename MortonCodeType, typename IndexType, typename FloatType>
    __global__
    void build_tree_kernel(const MortonCodeType* morton_codes, const IndexType num_mc, const IndexType num_bits,
        internal_node_t<MortonCodeType, IndexType, FloatType>* internal_nodes,
        IndexType* internal_children, MortonCodeType* internal_bounds,
        FloatType* internal_min_bounds, FloatType* internal_max_bounds)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_mc - 1) return;  // We have num_mc - 1 internal nodes

        IndexType first, last;
        morton_codes::determine_range<MortonCodeType, IndexType>(morton_codes, num_mc, tid, first, last, num_bits);
        IndexType delta_node = morton_codes::delta<MortonCodeType, IndexType>(morton_codes, num_mc, first, last, num_bits);
        IndexType split = morton_codes::find_split<MortonCodeType, IndexType>(morton_codes, num_mc, delta_node, first, last, num_bits);
        MortonCodeType lower_bound, upper_bound;
        morton_codes::find_lower_upper_bounds<MortonCodeType, IndexType>(morton_codes[split], delta_node, lower_bound, upper_bound, num_bits);

        IndexType num_bits_per_dim = num_bits / 3;
        MortonCodeType x_min, y_min, z_min;
        morton_codes::deinterleave_bits(lower_bound, num_bits_per_dim, x_min, y_min, z_min);
        MortonCodeType x_max, y_max, z_max;
        morton_codes::deinterleave_bits(upper_bound, num_bits_per_dim, x_max, y_max, z_max);

        FloatType size_per_dim = FloatType(1) / FloatType(1 << (num_bits_per_dim - 1));
        array3d_t<FloatType> min_bounds = {
            morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim),
            morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim),
            morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim)
        };
        array3d_t<FloatType> max_bounds = {
            morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_max, num_bits_per_dim) + size_per_dim,
            morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_max, num_bits_per_dim) + size_per_dim,
            morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_max, num_bits_per_dim) + size_per_dim
        };

        // Determine left and right children
        // n is the number of leaf nodes (num_mc)
        IndexType n = num_mc;
        IndexType left = (split == first) ? split : split + n;
        IndexType right = (split + 1 == last) ? split + 1 : split + 1 + n;

        internal_nodes[tid] = internal_node_t<MortonCodeType, IndexType, FloatType>{left, right, lower_bound, upper_bound, min_bounds, max_bounds};
        internal_children[tid * 2] = left;
        internal_children[tid * 2 + 1] = right;
        internal_bounds[tid * 2] = lower_bound;
        internal_bounds[tid * 2 + 1] = upper_bound;
        internal_min_bounds[tid * 3] = min_bounds[0];
        internal_min_bounds[tid * 3 + 1] = min_bounds[1];
        internal_min_bounds[tid * 3 + 2] = min_bounds[2];
        internal_max_bounds[tid * 3] = max_bounds[0];
        internal_max_bounds[tid * 3 + 1] = max_bounds[1];
        internal_max_bounds[tid * 3 + 2] = max_bounds[2];
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    __global__
    void compute_leaf_bounds_kernel(const MortonCodeType* morton_codes, const IndexType num_mc, const IndexType num_bits_per_dim,
        FloatType* min_bounds, FloatType* max_bounds)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= num_mc) return;

        MortonCodeType x_min, y_min, z_min;
        morton_codes::deinterleave_bits(morton_codes[tid], num_bits_per_dim, x_min, y_min, z_min);

        FloatType size_per_dim = FloatType(1) / FloatType(1 << (num_bits_per_dim - 1));
        FloatType x_min_f = morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(x_min, num_bits_per_dim);
        FloatType y_min_f = morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(y_min, num_bits_per_dim);
        FloatType z_min_f = morton_codes::morton_code_to_coordinate<FloatType, IndexType, MortonCodeType>(z_min, num_bits_per_dim);

        min_bounds[tid * 3] = x_min_f;
        min_bounds[tid * 3 + 1] = y_min_f;
        min_bounds[tid * 3 + 2] = z_min_f;

        max_bounds[tid * 3] = x_min_f + size_per_dim;
        max_bounds[tid * 3 + 1] = y_min_f + size_per_dim;
        max_bounds[tid * 3 + 2] = z_min_f + size_per_dim;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    cuda_binary_radix_tree_t<MortonCodeType, IndexType, FloatType>::cuda_binary_radix_tree_t(
        const vector<MortonCodeType>& h_morton_codes, const IndexType num_bits, cudaStream_t stream)
        : num_leaf_nodes(h_morton_codes.size()),
        internal_nodes(num_leaf_nodes - 1, stream),
        internal_children((num_leaf_nodes - 1) * 2, stream),
        internal_bounds((num_leaf_nodes - 1) * 2, stream),
        internal_min_bounds((num_leaf_nodes - 1) * 3, stream),
        internal_max_bounds((num_leaf_nodes - 1) * 3, stream),
        leaf_nodes(num_leaf_nodes, stream),
        leaf_min_bounds(num_leaf_nodes * 3, stream),
        leaf_max_bounds(num_leaf_nodes * 3, stream),
        internal_nodes_tex(0),
        internal_bounds_tex(0),
        internal_min_bounds_tex(0),
        internal_max_bounds_tex(0),
        leaf_nodes_tex(0),
        leaf_min_bounds_tex(0),
        leaf_max_bounds_tex(0)
    {
        assert(num_bits % 3 == 0);

        // copy leaf nodes
        cudaMemcpyAsync(leaf_nodes.data(), h_morton_codes.data(), num_leaf_nodes * sizeof(MortonCodeType), cudaMemcpyHostToDevice, stream);

        dim3 block_size(256, 1, 1), grid_size(1, 1, 1);
        grid_size.x = (num_leaf_nodes - 1 + block_size.x - 1) / block_size.x;
        build_tree_kernel<MortonCodeType, IndexType, FloatType><<<grid_size, block_size, 0, stream>>>(
            leaf_nodes.data(), num_leaf_nodes, num_bits, internal_nodes.data(),
            internal_children.data(), internal_bounds.data(),
            internal_min_bounds.data(), internal_max_bounds.data());

        IndexType num_bits_per_dim = num_bits / 3;
        grid_size.x = (num_leaf_nodes + block_size.x - 1) / block_size.x;
        compute_leaf_bounds_kernel<MortonCodeType, IndexType, FloatType><<<grid_size, block_size, 0, stream>>>(
            leaf_nodes.data(), num_leaf_nodes, num_bits_per_dim,
            leaf_min_bounds.data(), leaf_max_bounds.data());

        bind_texture_memory(internal_children.data(), (num_leaf_nodes - 1) * 2 * sizeof(IndexType), 32, cudaChannelFormatKindSigned, internal_nodes_tex);
        bind_texture_memory(internal_bounds.data(), (num_leaf_nodes - 1) * 2 * sizeof(MortonCodeType), 32, cudaChannelFormatKindUnsigned, internal_bounds_tex);
        bind_texture_memory(internal_min_bounds.data(), (num_leaf_nodes - 1) * 3 * sizeof(FloatType), 32, cudaChannelFormatKindFloat, internal_min_bounds_tex);
        bind_texture_memory(internal_max_bounds.data(), (num_leaf_nodes - 1) * 3 * sizeof(FloatType), 32, cudaChannelFormatKindFloat, internal_max_bounds_tex);
        bind_texture_memory(leaf_nodes.data(), num_leaf_nodes * sizeof(MortonCodeType), 32, cudaChannelFormatKindUnsigned, leaf_nodes_tex);
        bind_texture_memory(leaf_min_bounds.data(), num_leaf_nodes * 3 * sizeof(FloatType), 32, cudaChannelFormatKindFloat, leaf_min_bounds_tex);
        bind_texture_memory(leaf_max_bounds.data(), num_leaf_nodes * 3 * sizeof(FloatType), 32, cudaChannelFormatKindFloat, leaf_max_bounds_tex);
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    cuda_binary_radix_tree_t<MortonCodeType, IndexType, FloatType>::~cuda_binary_radix_tree_t()
    {
        unbind_texture_memory(internal_nodes_tex);
        unbind_texture_memory(internal_bounds_tex);
        unbind_texture_memory(internal_min_bounds_tex);
        unbind_texture_memory(internal_max_bounds_tex);
        unbind_texture_memory(leaf_nodes_tex);
        unbind_texture_memory(leaf_min_bounds_tex);
        unbind_texture_memory(leaf_max_bounds_tex);
    }

    template class cuda_binary_radix_tree_t<uint32_t, int32_t, gmp::gmp_float>;

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType, FloatType>::get_internal_nodes(vector<inode_t>& h_internal_nodes) const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        h_internal_nodes.resize(num_leaf_nodes - 1);
        cudaMemcpyAsync(h_internal_nodes.data(), internal_nodes.data(), (num_leaf_nodes - 1) * sizeof(inode_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    template <typename MortonCodeType, typename IndexType, typename FloatType>
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType, FloatType>::get_leaf_nodes(vector<MortonCodeType>& h_leaf_nodes) const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        h_leaf_nodes.resize(num_leaf_nodes);
        cudaMemcpyAsync(h_leaf_nodes.data(), leaf_nodes.data(), num_leaf_nodes * sizeof(MortonCodeType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType>
    __device__
    void cuda_tree_traverse(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t internal_bounds_tex,
        const cudaTextureObject_t internal_min_bounds_tex, const cudaTextureObject_t internal_max_bounds_tex,
        const cudaTextureObject_t leaf_nodes_tex, const cudaTextureObject_t leaf_min_bounds_tex, const cudaTextureObject_t leaf_max_bounds_tex,
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
                if constexpr (std::is_same_v<Checker, cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>>) {
                    IndexType bounds_base = node_index * 3;
                    array3d_t<FloatType> min_bounds = {
                        tex1Dfetch<FloatType>(leaf_min_bounds_tex, bounds_base),
                        tex1Dfetch<FloatType>(leaf_min_bounds_tex, bounds_base + 1),
                        tex1Dfetch<FloatType>(leaf_min_bounds_tex, bounds_base + 2)
                    };
                    array3d_t<FloatType> max_bounds = {
                        tex1Dfetch<FloatType>(leaf_max_bounds_tex, bounds_base),
                        tex1Dfetch<FloatType>(leaf_max_bounds_tex, bounds_base + 1),
                        tex1Dfetch<FloatType>(leaf_max_bounds_tex, bounds_base + 2)
                    };
                    check_method(min_bounds, max_bounds, position, cell_shift, indexes, num_indexes, indexes_offset);
                } else {
                    MortonCodeType morton_code = tex1Dfetch<MortonCodeType>(leaf_nodes_tex, node_index);
                    check_method(morton_code, node_index, position, cell_shift, indexes, num_indexes, indexes_offset);
                }
            } else {
                // Internal node
                IndexType metadata_base = (node_index - n) * 2;
                IndexType left = tex1Dfetch<IndexType>(internal_nodes_tex, metadata_base);
                IndexType right = tex1Dfetch<IndexType>(internal_nodes_tex, metadata_base + 1);
                bool should_traverse;
                if constexpr (std::is_same_v<Checker, cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>>) {
                    IndexType bounds_base = (node_index - n) * 3;
                    array3d_t<FloatType> min_bounds = {
                        tex1Dfetch<FloatType>(internal_min_bounds_tex, bounds_base),
                        tex1Dfetch<FloatType>(internal_min_bounds_tex, bounds_base + 1),
                        tex1Dfetch<FloatType>(internal_min_bounds_tex, bounds_base + 2)
                    };
                    array3d_t<FloatType> max_bounds = {
                        tex1Dfetch<FloatType>(internal_max_bounds_tex, bounds_base),
                        tex1Dfetch<FloatType>(internal_max_bounds_tex, bounds_base + 1),
                        tex1Dfetch<FloatType>(internal_max_bounds_tex, bounds_base + 2)
                    };
                    should_traverse = check_method(min_bounds, max_bounds, position, cell_shift);
                } else {
                    MortonCodeType lower_bound = tex1Dfetch<MortonCodeType>(internal_bounds_tex, metadata_base);
                    MortonCodeType upper_bound = tex1Dfetch<MortonCodeType>(internal_bounds_tex, metadata_base + 1);
                    should_traverse = check_method(lower_bound, upper_bound, position, cell_shift);
                }

                if (should_traverse) {
                    if (stack_top < 63) stack_data[++stack_top] = left;
                    if (stack_top < 63) stack_data[++stack_top] = right;
                }
            }
        }
        return;
    }

    template __device__
    void cuda_tree_traverse<cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t>
    (const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t internal_bounds_tex,
        const cudaTextureObject_t internal_min_bounds_tex, const cudaTextureObject_t internal_max_bounds_tex,
        const cudaTextureObject_t leaf_nodes_tex, const cudaTextureObject_t leaf_min_bounds_tex, const cudaTextureObject_t leaf_max_bounds_tex,
        const int32_t num_leaf_nodes,
        const cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t> check_method,
        const point3d_t<gmp::gmp_float> position, const array3d_t<int32_t> cell_shift, int32_t* indexes, int32_t& num_indexes, const int32_t indexes_offset);

    template __device__
    void cuda_tree_traverse<cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t>
    (const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t internal_bounds_tex,
        const cudaTextureObject_t internal_min_bounds_tex, const cudaTextureObject_t internal_max_bounds_tex,
        const cudaTextureObject_t leaf_nodes_tex, const cudaTextureObject_t leaf_min_bounds_tex, const cudaTextureObject_t leaf_max_bounds_tex,
        const int32_t num_leaf_nodes,
        const cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t> check_method,
        const point3d_t<gmp::gmp_float> position, const array3d_t<int32_t> cell_shift, int32_t* indexes, int32_t& num_indexes, const int32_t indexes_offset);

    void bind_texture_memory(void* data_ptr, uint32_t size, int bits_per_channel, cudaChannelFormatKind format, cudaTextureObject_t& tex)
    {
        if (size == 0 || data_ptr == nullptr) {
            tex = 0;
            return;
        }
        // Create texture descriptor for internal nodes
        cudaResourceDesc resDesc_internal = {};
        resDesc_internal.resType = cudaResourceTypeLinear;
        resDesc_internal.res.linear.devPtr = data_ptr;
        resDesc_internal.res.linear.desc.f = format;
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
        if (tex) {
            cudaDestroyTextureObject(tex);
        }
    }


    template <class Checker, typename MortonCodeType, typename FloatType, typename IndexType, int MAX_STACK>
    __global__
    void cuda_tree_traverse_warp(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t internal_bounds_tex,
        const cudaTextureObject_t internal_min_bounds_tex, const cudaTextureObject_t internal_max_bounds_tex,
        const cudaTextureObject_t leaf_nodes_tex, const cudaTextureObject_t leaf_min_bounds_tex, const cudaTextureObject_t leaf_max_bounds_tex, const IndexType num_leaf_nodes,
        const Checker check_method, const point3d_t<FloatType>* positions, const IndexType* query_target_indexes,
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
                if constexpr (std::is_same_v<Checker, cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>>) {
                    FloatType min_x = FloatType(0), min_y = FloatType(0), min_z = FloatType(0);
                    FloatType max_x = FloatType(0), max_y = FloatType(0), max_z = FloatType(0);
                    if (lane == 0) {
                        IndexType bounds_base = node_index * 3;
                        min_x = tex1Dfetch<FloatType>(leaf_min_bounds_tex, bounds_base);
                        min_y = tex1Dfetch<FloatType>(leaf_min_bounds_tex, bounds_base + 1);
                        min_z = tex1Dfetch<FloatType>(leaf_min_bounds_tex, bounds_base + 2);
                        max_x = tex1Dfetch<FloatType>(leaf_max_bounds_tex, bounds_base);
                        max_y = tex1Dfetch<FloatType>(leaf_max_bounds_tex, bounds_base + 1);
                        max_z = tex1Dfetch<FloatType>(leaf_max_bounds_tex, bounds_base + 2);
                    }
                    min_x = __shfl_sync(0xffffffff, min_x, 0);
                    min_y = __shfl_sync(0xffffffff, min_y, 0);
                    min_z = __shfl_sync(0xffffffff, min_z, 0);
                    max_x = __shfl_sync(0xffffffff, max_x, 0);
                    max_y = __shfl_sync(0xffffffff, max_y, 0);
                    max_z = __shfl_sync(0xffffffff, max_z, 0);
                    if (!lane_active) continue;
                    array3d_t<FloatType> min_bounds = {min_x, min_y, min_z};
                    array3d_t<FloatType> max_bounds = {max_x, max_y, max_z};
                    check_method(min_bounds, max_bounds, position, cell_shift, indexes, num_index, indexes_offset);
                } else {
                    if (!lane_active) continue;
                    check_method(morton_code, node_index, position, cell_shift, indexes, num_index, indexes_offset);
                }
            } else {
                IndexType metadata_base = (node_index - num_leaf_nodes) * 2;
                MortonCodeType lower_bound = lane == 0 ? tex1Dfetch<MortonCodeType>(internal_bounds_tex, metadata_base) : 0;
                MortonCodeType upper_bound = lane == 0 ? tex1Dfetch<MortonCodeType>(internal_bounds_tex, metadata_base + 1) : 0;
                IndexType left = lane == 0 ? tex1Dfetch<IndexType>(internal_nodes_tex, metadata_base) : 0;
                IndexType right = lane == 0 ? tex1Dfetch<IndexType>(internal_nodes_tex, metadata_base + 1) : 0;
                lower_bound = __shfl_sync(0xffffffff, lower_bound, 0);
                upper_bound = __shfl_sync(0xffffffff, upper_bound, 0);
                left = __shfl_sync(0xffffffff, left, 0);
                right = __shfl_sync(0xffffffff, right, 0);

                bool hit;
                if constexpr (std::is_same_v<Checker, cuda_check_sphere_t<MortonCodeType, FloatType, IndexType>>) {
                    FloatType min_x = FloatType(0), min_y = FloatType(0), min_z = FloatType(0);
                    FloatType max_x = FloatType(0), max_y = FloatType(0), max_z = FloatType(0);
                    if (lane == 0) {
                        IndexType bounds_base = (node_index - num_leaf_nodes) * 3;
                        min_x = tex1Dfetch<FloatType>(internal_min_bounds_tex, bounds_base);
                        min_y = tex1Dfetch<FloatType>(internal_min_bounds_tex, bounds_base + 1);
                        min_z = tex1Dfetch<FloatType>(internal_min_bounds_tex, bounds_base + 2);
                        max_x = tex1Dfetch<FloatType>(internal_max_bounds_tex, bounds_base);
                        max_y = tex1Dfetch<FloatType>(internal_max_bounds_tex, bounds_base + 1);
                        max_z = tex1Dfetch<FloatType>(internal_max_bounds_tex, bounds_base + 2);
                    }
                    min_x = __shfl_sync(0xffffffff, min_x, 0);
                    min_y = __shfl_sync(0xffffffff, min_y, 0);
                    min_z = __shfl_sync(0xffffffff, min_z, 0);
                    max_x = __shfl_sync(0xffffffff, max_x, 0);
                    max_y = __shfl_sync(0xffffffff, max_y, 0);
                    max_z = __shfl_sync(0xffffffff, max_z, 0);
                    array3d_t<FloatType> min_bounds = {min_x, min_y, min_z};
                    array3d_t<FloatType> max_bounds = {max_x, max_y, max_z};
                    hit = lane_active && check_method(min_bounds, max_bounds, position, cell_shift);
                } else {
                    hit = lane_active && check_method(lower_bound, upper_bound, position, cell_shift);
                }
                IndexType parent_mask  = __ballot_sync(0xffffffff, hit);

                if (lane == 0 && parent_mask) {
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
    (const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t internal_bounds_tex,
        const cudaTextureObject_t internal_min_bounds_tex, const cudaTextureObject_t internal_max_bounds_tex,
        const cudaTextureObject_t leaf_nodes_tex, const cudaTextureObject_t leaf_min_bounds_tex, const cudaTextureObject_t leaf_max_bounds_tex, const int32_t num_leaf_nodes,
        const cuda_check_intersect_box_t<uint32_t, gmp::gmp_float, int32_t> check_method,
        const point3d_t<gmp::gmp_float>* positions, const int32_t* query_target_indexes, const array3d_t<int32_t>* cell_shifts, const int32_t num_queries,
        int32_t* indexes, int32_t* num_indexes, const int32_t* num_indexes_offset);

    template __global__
    void cuda_tree_traverse_warp<cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t>, uint32_t, gmp::gmp_float, int32_t, 24>
    (const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t internal_bounds_tex,
        const cudaTextureObject_t internal_min_bounds_tex, const cudaTextureObject_t internal_max_bounds_tex,
        const cudaTextureObject_t leaf_nodes_tex, const cudaTextureObject_t leaf_min_bounds_tex, const cudaTextureObject_t leaf_max_bounds_tex, const int32_t num_leaf_nodes,
        const cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t> check_method,
        const point3d_t<gmp::gmp_float>* positions, const int32_t* query_target_indexes, const array3d_t<int32_t>* cell_shifts, const int32_t num_queries,
        int32_t* indexes, int32_t* num_indexes, const int32_t* num_indexes_offset);
        
}} // namespace gmp::tree 