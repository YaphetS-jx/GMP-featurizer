#include "cuda_tree.hpp"
#include <cuda_runtime.h>

namespace gmp { namespace tree {

    // cuda_binary_radix_tree_t implementation
    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container_host, template<typename, typename...> class Container_device>
    cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::cuda_binary_radix_tree_t(const morton_container_device_t& morton_codes, const IndexType num_bits, cudaStream_t stream)
        : internal_nodes(morton_codes.size() - 1, stream),
          leaf_nodes(morton_codes.size(), stream),
          internal_nodes_tex(0),
          leaf_nodes_tex(0),
          textures_bound(false)
    {
        build_tree(morton_codes, num_bits, stream);
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container_host, template<typename, typename...> class Container_device>
    cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::~cuda_binary_radix_tree_t()
    {
        unbind_texture_memory();
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container_host, template<typename, typename...> class Container_device>
    typename cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::node_container_host_t
    cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::get_internal_nodes() const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        node_container_host_t internal_nodes_host;
        internal_nodes_host.reserve(internal_nodes.size());
        
        // Create a temporary buffer for the data
        auto temp_buffer = static_cast<internal_node_t<MortonCodeType, IndexType>*>(malloc(internal_nodes.size() * sizeof(internal_node_t<MortonCodeType, IndexType>)));
        cudaMemcpyAsync(temp_buffer, internal_nodes.data(), 
            internal_nodes.size() * sizeof(internal_node_t<MortonCodeType, IndexType>), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Copy the data to the host vector
        for (size_t i = 0; i < internal_nodes.size(); ++i) {
            internal_nodes_host.push_back(temp_buffer[i]);
        }
        
        free(temp_buffer);
        return internal_nodes_host;
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container_host, template<typename, typename...> class Container_device>
    typename cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::morton_container_host_t
    cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::get_leaf_nodes() const
    {
        auto stream = gmp::resources::gmp_resource::instance().get_stream();
        morton_container_host_t leaf_nodes_host;
        leaf_nodes_host.reserve(leaf_nodes.size());
        
        // Create a temporary buffer for the data
        auto temp_buffer = static_cast<MortonCodeType*>(malloc(leaf_nodes.size() * sizeof(MortonCodeType)));
        cudaMemcpyAsync(temp_buffer, leaf_nodes.data(), 
            leaf_nodes.size() * sizeof(MortonCodeType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Copy the data to the host vector
        for (size_t i = 0; i < leaf_nodes.size(); ++i) {
            leaf_nodes_host.push_back(temp_buffer[i]);
        }
        
        free(temp_buffer);
        return leaf_nodes_host;
    }

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

        internal_nodes[tid] = internal_node_t<MortonCodeType, IndexType>(left, right, lower_bound, upper_bound);
    }

    // CUDA kernels for tree traversal
    template <typename MortonCodeType, typename IndexType>
    __global__
    void count_results_kernel(cudaTextureObject_t internal_nodes_tex,
                             cudaTextureObject_t leaf_nodes_tex,
                             const IndexType num_leaf_nodes,
                             const MortonCodeType query_lower_bound,
                             const MortonCodeType query_upper_bound,
                             IndexType* result_counts)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= 1) return;  // Only one thread does the counting
        
        // Fixed stack for traversal
        IndexType stack_data[64];
        int stack_top = -1;
        
        IndexType count = 0;
        IndexType n = num_leaf_nodes;
        
        // Start with root (internal node index n)
        stack_data[++stack_top] = n;
        
        while (stack_top >= 0) {
            IndexType node_index = stack_data[stack_top--];
            
            if (node_index < n) {
                // Leaf node
                MortonCodeType morton_code = tex1Dfetch<MortonCodeType>(leaf_nodes_tex, node_index);
                
                // Check if morton code is within query bounds
                MortonCodeType x_mask, y_mask, z_mask;
                morton_codes::create_masks(x_mask, y_mask, z_mask);
                
                if (morton_codes::mc_is_less_than_or_equal(query_lower_bound, morton_code, x_mask, y_mask, z_mask) && 
                    morton_codes::mc_is_less_than_or_equal(morton_code, query_upper_bound, x_mask, y_mask, z_mask)) {
                    count++;
                }
            } else {
                // Internal node
                IndexType left = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - n) * 4);
                IndexType right = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - n) * 4 + 1);
                MortonCodeType lower_bound = tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - n) * 4 + 2);
                MortonCodeType upper_bound = tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - n) * 4 + 3);
                
                internal_node_t<MortonCodeType, IndexType> node(left, right, lower_bound, upper_bound);
                
                // Check if node bounds intersect with query bounds
                MortonCodeType x_mask, y_mask, z_mask;
                morton_codes::create_masks(x_mask, y_mask, z_mask);
                
                if (morton_codes::mc_is_less_than_or_equal(query_lower_bound, node.upper_bound, x_mask, y_mask, z_mask) && 
                    morton_codes::mc_is_less_than_or_equal(node.lower_bound, query_upper_bound, x_mask, y_mask, z_mask)) {
                    
                    if (stack_top < 63) stack_data[++stack_top] = node.left;
                    if (stack_top < 63) stack_data[++stack_top] = node.right;
                }
            }
        }
        
        result_counts[0] = count;
    }

    template <typename MortonCodeType, typename IndexType>
    __global__
    void collect_results_kernel(cudaTextureObject_t internal_nodes_tex,
                               cudaTextureObject_t leaf_nodes_tex,
                               const IndexType num_leaf_nodes,
                               const MortonCodeType query_lower_bound,
                               const MortonCodeType query_upper_bound,
                               IndexType* result_indices,
                               int32_t* result_shifts)
    {
        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= 1) return;  // Only one thread does the collection
        
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
                MortonCodeType x_mask, y_mask, z_mask;
                morton_codes::create_masks(x_mask, y_mask, z_mask);
                
                if (morton_codes::mc_is_less_than_or_equal(query_lower_bound, morton_code, x_mask, y_mask, z_mask) && 
                    morton_codes::mc_is_less_than_or_equal(morton_code, query_upper_bound, x_mask, y_mask, z_mask)) {
                    
                    result_indices[result_index] = node_index;
                    // Store shifts as 3 consecutive int32s
                    result_shifts[result_index * 3] = 0;
                    result_shifts[result_index * 3 + 1] = 0;
                    result_shifts[result_index * 3 + 2] = 0;
                    result_index++;
                }
            } else {
                // Internal node
                IndexType left = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - n) * 4);
                IndexType right = tex1Dfetch<IndexType>(internal_nodes_tex, (node_index - n) * 4 + 1);
                MortonCodeType lower_bound = tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - n) * 4 + 2);
                MortonCodeType upper_bound = tex1Dfetch<MortonCodeType>(internal_nodes_tex, (node_index - n) * 4 + 3);
                
                internal_node_t<MortonCodeType, IndexType> node(left, right, lower_bound, upper_bound);
                
                // Check if node bounds intersect with query bounds
                MortonCodeType x_mask, y_mask, z_mask;
                morton_codes::create_masks(x_mask, y_mask, z_mask);
                
                if (morton_codes::mc_is_less_than_or_equal(query_lower_bound, node.upper_bound, x_mask, y_mask, z_mask) && 
                    morton_codes::mc_is_less_than_or_equal(node.lower_bound, query_upper_bound, x_mask, y_mask, z_mask)) {
                    
                    if (stack_top < 63) stack_data[++stack_top] = node.left;
                    if (stack_top < 63) stack_data[++stack_top] = node.right;
                }
            }
        }
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container_host, template<typename, typename...> class Container_device>
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::
        build_tree(const morton_container_device_t& morton_codes, const IndexType num_bits, cudaStream_t stream)
    {
        assert(num_bits % 3 == 0);
        auto num_mc = static_cast<IndexType>(morton_codes.size());
        internal_nodes.resize(num_mc - 1, stream);
        leaf_nodes.resize(num_mc, stream);

        dim3 block_size(256, 1, 1), grid_size(1, 1, 1);
        grid_size.x = (num_mc - 1 + block_size.x - 1) / block_size.x;
        build_tree_kernel<MortonCodeType, IndexType><<<grid_size, block_size, 0, stream>>>(morton_codes.data(), num_mc, num_bits, internal_nodes.data());
        
        // copy leaf nodes
        cudaMemcpyAsync(leaf_nodes.data(), morton_codes.data(), num_mc * sizeof(MortonCodeType), cudaMemcpyDeviceToDevice, stream);
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container_host, template<typename, typename...> class Container_device>
    typename cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::traverse_result_t
    cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::
        traverse(const MortonCodeType query_lower_bound, const MortonCodeType query_upper_bound, cudaStream_t stream) const
    {
        // Ensure textures are bound
        if (!textures_bound) {
            const_cast<cuda_binary_radix_tree_t*>(this)->bind_texture_memory(stream);
        }
        
        IndexType num_leaf_nodes = static_cast<IndexType>(leaf_nodes.size());
        
        // First pass: count results
        vector_device<IndexType> result_counts(1, stream);
        dim3 block_size(1, 1, 1), grid_size(1, 1, 1);
        
        count_results_kernel<MortonCodeType, IndexType><<<grid_size, block_size, 0, stream>>>(
            internal_nodes_tex, leaf_nodes_tex, num_leaf_nodes, 
            query_lower_bound, query_upper_bound, result_counts.data());
        
        // Get count from device
        IndexType count;
        cudaMemcpyAsync(&count, result_counts.data(), sizeof(IndexType), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Second pass: collect results
        vector_device<IndexType> result_indices(count, stream);
        vector_device<int32_t> result_shifts(count * 3, stream); // 3 int32s per result
        
        collect_results_kernel<MortonCodeType, IndexType><<<grid_size, block_size, 0, stream>>>(
            internal_nodes_tex, leaf_nodes_tex, num_leaf_nodes,
            query_lower_bound, query_upper_bound, 
            result_indices.data(), result_shifts.data());
        
        traverse_result_t result{std::move(result_indices), std::move(result_shifts)};
        return result;
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container_host, template<typename, typename...> class Container_device>
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::bind_texture_memory(cudaStream_t stream)
    {
        if (textures_bound) return;
        
        // Create texture descriptor for internal nodes
        cudaResourceDesc resDesc_internal = {};
        resDesc_internal.resType = cudaResourceTypeLinear;
        resDesc_internal.res.linear.devPtr = internal_nodes.data();
        resDesc_internal.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        resDesc_internal.res.linear.desc.x = 32; // bits per channel
        resDesc_internal.res.linear.sizeInBytes = internal_nodes.size() * sizeof(internal_node_t<MortonCodeType, IndexType>);
        
        cudaTextureDesc texDesc = {};
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.addressMode[0] = cudaAddressModeClamp;
        
        cudaCreateTextureObject(&internal_nodes_tex, &resDesc_internal, &texDesc, nullptr);
        
        // Create texture descriptor for leaf nodes
        cudaResourceDesc resDesc_leaf = {};
        resDesc_leaf.resType = cudaResourceTypeLinear;
        resDesc_leaf.res.linear.devPtr = leaf_nodes.data();
        resDesc_leaf.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        resDesc_leaf.res.linear.desc.x = 32; // bits per channel
        resDesc_leaf.res.linear.sizeInBytes = leaf_nodes.size() * sizeof(MortonCodeType);
        
        cudaCreateTextureObject(&leaf_nodes_tex, &resDesc_leaf, &texDesc, nullptr);
        
        textures_bound = true;
    }

    template <typename MortonCodeType, typename IndexType, template<typename, typename...> class Container_host, template<typename, typename...> class Container_device>
    void cuda_binary_radix_tree_t<MortonCodeType, IndexType, Container_host, Container_device>::unbind_texture_memory()
    {
        if (!textures_bound) return;
        
        if (internal_nodes_tex) {
            cudaDestroyTextureObject(internal_nodes_tex);
            internal_nodes_tex = 0;
        }
        
        if (leaf_nodes_tex) {
            cudaDestroyTextureObject(leaf_nodes_tex);
            leaf_nodes_tex = 0;
        }
        
        textures_bound = false;
    }

    // Explicit instantiations for cuda_binary_radix_tree_t (used in tests)
    template class cuda_binary_radix_tree_t<uint32_t, int32_t>;

}} // namespace gmp::tree 