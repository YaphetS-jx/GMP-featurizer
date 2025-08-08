#include "cuda_tree.hpp"
#include "resources.hpp"
#include "cuda_thrust_ops.hpp"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include "cuda_util.hpp"

namespace gmp { namespace tree {

    using namespace gmp::thrust_ops;

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
        const vector_device<MortonCodeType>& morton_codes, const IndexType num_bits, cudaStream_t stream)
        : num_leaf_nodes(morton_codes.size()), 
        internal_nodes(num_leaf_nodes - 1, stream), 
        leaf_nodes(num_leaf_nodes, stream)
    {
        assert(num_bits % 3 == 0);

        dim3 block_size(256, 1, 1), grid_size(1, 1, 1);
        grid_size.x = (num_leaf_nodes - 1 + block_size.x - 1) / block_size.x;
        build_tree_kernel<MortonCodeType, IndexType><<<grid_size, block_size, 0, stream>>>(morton_codes.data(), num_leaf_nodes, num_bits, internal_nodes.data());
        
        // copy leaf nodes
        cudaMemcpyAsync(leaf_nodes.data(), morton_codes.data(), num_leaf_nodes * sizeof(MortonCodeType), cudaMemcpyDeviceToDevice, stream);

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
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType>::get_internal_nodes(vector_host<inode_t>& h_internal_nodes) const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        h_internal_nodes.resize(num_leaf_nodes - 1);
        cudaMemcpyAsync(h_internal_nodes.data(), internal_nodes.data(), (num_leaf_nodes - 1) * sizeof(inode_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    template <typename MortonCodeType, typename IndexType>
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType>::get_leaf_nodes(vector_host<MortonCodeType>& h_leaf_nodes) const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        h_leaf_nodes.resize(num_leaf_nodes);
        cudaMemcpyAsync(h_leaf_nodes.data(), leaf_nodes.data(), num_leaf_nodes * sizeof(MortonCodeType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    template <typename MortonCodeType, typename IndexType>
    __device__
    void tree_traverse_kernel(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t leaf_nodes_tex, const IndexType num_leaf_nodes, 
        const cuda_compare_op_t<MortonCodeType, IndexType>& check_method, const array3d_t<IndexType>& cell_shifts,
        IndexType* indexes, size_t* num_indexes)
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
                check_method(morton_code, cell_shifts, node_index, indexes, num_indexes);
            } else {
                // Internal node
                IndexType left = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - n) * 4);
                IndexType right = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - n) * 4 + 1);
                MortonCodeType lower_bound = tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - n) * 4 + 2);
                MortonCodeType upper_bound = tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - n) * 4 + 3);
                
                if (check_method(lower_bound, upper_bound, cell_shifts)) {
                    if (stack_top < 63) stack_data[++stack_top] = left;
                    if (stack_top < 63) stack_data[++stack_top] = right;
                }
            }
        }
        return;
    }
    
    template __device__
    void tree_traverse_kernel<uint32_t, int32_t>(cudaTextureObject_t, cudaTextureObject_t,
        int32_t, const cuda_compare_op_t<uint32_t, int32_t>&, const array3d_t<int32_t>&, int32_t*, size_t*);

    void bind_texture_memory(void* data_ptr, size_t size, int bits_per_channel, cudaTextureObject_t& tex)
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

}} // namespace gmp::tree 