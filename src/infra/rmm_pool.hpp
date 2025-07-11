#include <map>
#include <thread>
#include <mutex>
#include <memory>
#include <cuda_runtime.h>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include "memory_manager.hpp"

#define IROUNDUP(x, y) (((x) + (y) - 1) & ~((y) - 1))

namespace gmp { namespace resources {

    class StreamedDeviceMemory {
    public:
        static size_t prefixsum_aligned_n_chunks(const size_t sizes[], size_t n, size_t prefix[]);
        StreamedDeviceMemory(std::mutex* mutex, rmm::mr::cuda_async_memory_resource* dmmr);
        ~StreamedDeviceMemory() = default;

        void* allocate(size_t size);
        void* safe_allocate(size_t size);
        void allocate_n(void* ptrs[], const size_t sizes[], size_t n);
        void deallocate(void* ptr);
        void deallocate_n(void* ptrs[], size_t n);
        void insert_thread_stream_mapping(pthread_t thread_key, cudaStream_t stream);
        void print_map();
        size_t allocate_n_chunks(void* chunks[], const size_t sizes[], size_t n);
        size_t allocate_n_chunks(void* chunks[], const size_t sizes[], size_t n,  size_t prefixsum[]);

    private:
        std::mutex* _mutex;
        rmm::mr::cuda_async_memory_resource* _dmmr;
        std::map<pthread_t, cudaStream_t> _thread_stream_map;
    };

    class StreamedMemory
    {
    public:
        StreamedMemory(std::shared_ptr<std::mutex> mutex, std::shared_ptr<rmm::mr::cuda_async_memory_resource> dmmr);
        ~StreamedMemory() = default;

        void* allocate(size_t size, cudaStream_t stream);
        void deallocate(void* ptr, cudaStream_t stream);

    private:
        std::shared_ptr<std::mutex> _mutex;
        std::shared_ptr<rmm::mr::cuda_async_memory_resource> _dmmr;
    };

    class PinnedMemory
    {
    public:
        static size_t prefixsum_aligned_n_chunks(const size_t sizes[], size_t n, size_t prefix[]);
        PinnedMemory(block_t* head, size_t alignment, std::mutex* mutex);
        void* allocate(size_t size);
        void* safe_allocate(size_t size);
        void deallocate(void* ptr);
        void allocate_n(void* ptrs[], const size_t sizes[], size_t n);
        void deallocate_n(void* ptrs[], size_t n);
        size_t allocate_n_chunks(void* chunks[], const size_t sizes[], size_t n);
        size_t allocate_n_chunks(void* chunks[], const size_t sizes[], size_t n,  size_t prefixsum[]);
        void printBlocks();

    private:
        block_t* _head;
        size_t _alignment;
        std::mutex* _mutex;
    };

    class PinnedHostMemory
    {
    public:
        PinnedHostMemory(std::shared_ptr<std::mutex> mutex, block_t* head, size_t alignment);
        void* allocate(size_t size);
        void deallocate(void* ptr);

    private:
        block_t* _head;
        size_t _alignment;
        std::shared_ptr<std::mutex> _mutex;
    };

    class ManagedMemory
    {
    public:
        ManagedMemory(block_t* head, size_t alignment, std::mutex* mutex);
        void* allocate(size_t size);
        void* safe_allocate(size_t size);
        void deallocate(void* ptr);

    private:
        block_t* _head;
        size_t _alignment;
        std::mutex* _mutex;
    };

}} // namespace gmp::resources