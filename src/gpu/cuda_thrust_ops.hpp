#pragma once
#include <thrust/device_malloc_allocator.h>
#include "resources.hpp"
#include "containers.hpp"
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>      // upper_bound
#include <thrust/copy.h>               // copy_if
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

namespace gmp { namespace thrust_ops {

#ifndef THRUST_CALL
    #define THRUST_CALL(FUNC, allocator, stream, ...) \
    FUNC(thrust::cuda::par(ThrustAllocator<char>(allocator, stream)).on(stream), __VA_ARGS__)
#endif

    // Refer to the following link: https://stackoverflow.com/questions/76594790/memory-pool-in-thrust-execution-policy

    template <typename T>
    struct ThrustAllocator : public thrust::device_malloc_allocator<T>
    {
        using Base = thrust::device_malloc_allocator<T>;
        using pointer = typename Base::pointer;
        using size_type = typename Base::size_type;

        public:
        ThrustAllocator(gmp::resources::device_memory_manager* dm, cudaStream_t stream)
            : _dm(dm), _stream(stream) {}

        pointer allocate(size_type n)
        {
            return thrust::device_pointer_cast((T*)_dm->allocate(n * sizeof(T), _stream));
        }

        void deallocate(pointer p, size_type n)
        {
            _dm->deallocate(thrust::raw_pointer_cast(p), _stream);
        }

    private:
        gmp::resources::device_memory_manager* _dm;
        cudaStream_t _stream;
    };


    using gmp::containers::vector_device;

    // get index mapping from offset 
    template <typename IndexType>
    void get_index_mapping(const vector_device<IndexType>& inclusive_offset, vector_device<IndexType>& index_mapping, 
        gmp::resources::device_memory_manager* dm, cudaStream_t stream)
    {
        auto idx0 = thrust::make_counting_iterator<IndexType>(0);
        THRUST_CALL(thrust::upper_bound, dm, stream, 
            inclusive_offset.begin(), inclusive_offset.end(), idx0, idx0 + index_mapping.size(), index_mapping.begin());
    }

    template <typename IndexType>
    void get_index_mapping(const vector_device<IndexType>& inclusive_offset, const vector_device<IndexType>& values,
        vector_device<IndexType>& index_mapping, 
        gmp::resources::device_memory_manager* dm, cudaStream_t stream)
    {   
        THRUST_CALL(thrust::upper_bound, dm, stream, 
            inclusive_offset.begin(), inclusive_offset.end(), values.begin(), values.end(), index_mapping.begin());
    }

    // values: concatenated array of length N
    // ends_inclusive: size M, inclusive ends (e.g., lengths [3,2,4] -> ends [3,5,9])
    template <typename T, typename Index, typename Operator>
    void segmented_sort_inplace(vector_device<T>& values, const vector_device<Index>& ends_inclusive, Operator op, 
        gmp::resources::device_memory_manager* dm, cudaStream_t stream) 
    {
        const Index N = static_cast<Index>(values.size());
        const Index M = static_cast<Index>(ends_inclusive.size());
        if (N == 0 || M == 0) return;

        // Build seg_id[i] = index of the segment that element i belongs to
        vector_device<Index> seg_id(N, stream);
        get_index_mapping(ends_inclusive, seg_id, dm, stream);
        // Now sort (seg_id, value) lexicographically
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(seg_id.data(), values.data()));
        auto zip_end   = thrust::make_zip_iterator(thrust::make_tuple(seg_id.data() + N,   values.data() + N));
        // sort the values within each segment
        THRUST_CALL(thrust::sort, dm, stream, zip_begin, zip_end, op);
        // `values` is now sorted within each segment, segments remain in original order.
    }

    // Generic compact function using thrust::copy_if
    template <typename IndexType, typename FlagType, typename CompOp>
    IndexType compact(const vector_device<FlagType>& flags, 
        vector_device<IndexType>& output_indices, CompOp comp_op,
        gmp::resources::device_memory_manager* dm, cudaStream_t stream)
    {
        vector_device<IndexType> idx(flags.size(), stream);
        THRUST_CALL(thrust::sequence, dm, stream, idx.begin(), idx.end());
        
        // Perform the copy_if operation
        auto end_it = THRUST_CALL(thrust::copy_if, dm, stream,
            idx.begin(), idx.end(), flags.begin(), output_indices.begin(), comp_op);
        
        // Return the number of selected elements
        return static_cast<IndexType>(end_it - output_indices.begin());
    }

    // Functor for non-zero check (instead of lambda to avoid --extended-lambda requirement)
    template <typename FlagType>
    struct non_zero_predicate {
        __host__ __device__
        bool operator()(FlagType f) const {
            return f != 0;
        }
    };

    template <typename IndexType, typename FlagType>
    IndexType compact(const vector_device<FlagType>& flags, 
        vector_device<IndexType>& output_indices,
        gmp::resources::device_memory_manager* dm, cudaStream_t stream)
    {
        non_zero_predicate<FlagType> comp_op;
        return compact(flags, output_indices, comp_op, dm, stream);
    }


}} // namespace gmp::thrust_ops