#include <thread>
#include <iostream>
#include <unistd.h>

#include "rmm_pool.hpp"
#include "memory_manager.hpp"

namespace gmp { namespace resources {

    const int GPU_MEMORY_ALIGNMENT = 256;

    StreamedDeviceMemory::StreamedDeviceMemory(std::mutex* mutex, rmm::mr::cuda_async_memory_resource* dmmr) : _mutex(mutex), _dmmr(dmmr)
    {
        if (_mutex == nullptr || _dmmr == nullptr)
        {
            throw std::runtime_error("Invalid arguments passed to StreamedDeviceMemory constructor.");
        }
    }

    size_t StreamedDeviceMemory::prefixsum_aligned_n_chunks(const size_t sizes[], size_t n, size_t prefix[])
    {
        if (n == 0) return 0;

        const int my_alignment = GPU_MEMORY_ALIGNMENT;

        prefix[0] = 0;
        for (size_t i = 0; i < n - 1; ++i) {
            prefix[i + 1] = prefix[i] + IROUNDUP(sizes[i], my_alignment);
        }

        return prefix[n - 1] + sizes[n - 1];
    }

    void* StreamedDeviceMemory::allocate(size_t size)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        pthread_t thread_key = pthread_self();
        void *ptr = nullptr;
        auto it = _thread_stream_map.find(thread_key);
        assert(it != _thread_stream_map.end());
        ptr = _dmmr->allocate(size, rmm::cuda_stream_view(it->second));
        return ptr;
    }

    size_t StreamedDeviceMemory::allocate_n_chunks(void* chunks[], const size_t sizes[], size_t n)
    {
        if (n == 0) return 0;

        const int my_alignment = GPU_MEMORY_ALIGNMENT;

        size_t total_size = 0;
        for (size_t i = 0; i < n; ++i) {
            total_size += IROUNDUP(sizes[i], my_alignment);
        }

        void* base_ptr = allocate(total_size);  // Uses thread-local stream

        size_t offset = 0;
        for (size_t i = 0; i < n; ++i) {
            chunks[i] = (sizes[i] != 0) ? static_cast<char*>(base_ptr) + offset : nullptr;
            offset += IROUNDUP(sizes[i], my_alignment);
        }

        // Ensure the first chunk points to the base allocation
        chunks[0] = base_ptr;

        return total_size;
    }

    size_t StreamedDeviceMemory::allocate_n_chunks(void* chunks[], const size_t sizes[], size_t n,  size_t prefixsum[])
    {
        prefixsum[0] = 0;
        for (size_t i = 0; i < n - 1; ++i) {
            prefixsum[i + 1] = prefixsum[i] + IROUNDUP(sizes[i], GPU_MEMORY_ALIGNMENT);
        }

        size_t sum = prefixsum[n - 1] + sizes[n - 1];
        void* ptr = allocate(sum);

        for (size_t i = 0; i < n; ++i) {
            chunks[i] = (sizes[i] != 0) ? ((char*)ptr + prefixsum[i]) : NULL;
        }

        // 0th chunk to point to the allocated large buffer, and be non-NULL
        chunks[0] = ptr;
        return sum;
    }

    void* StreamedDeviceMemory::safe_allocate(size_t size)
    {
        while (true)
        {
            try
            {
                std::lock_guard<std::mutex> guard(*_mutex);
                pthread_t thread_key = pthread_self();
                auto it = _thread_stream_map.find(thread_key);
                assert(it != _thread_stream_map.end());
                void* ptr = _dmmr->allocate(size, rmm::cuda_stream_view(it->second));
                return ptr;
            }
            catch (rmm::bad_alloc const& ex)
            {
                // Introduce a small delay before retrying
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    void StreamedDeviceMemory::deallocate(void* ptr)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        pthread_t thread_key = pthread_self();
        auto it = _thread_stream_map.find(thread_key);
        // assert(it != _thread_stream_map.end());
        if (it == _thread_stream_map.end())
        {
            _dmmr->deallocate(ptr, 0, rmm::cuda_stream_view(cudaStream_t(0)));
        }
        else
        {
            _dmmr->deallocate(ptr, 0, rmm::cuda_stream_view(it->second));
        }
    }

    void StreamedDeviceMemory::print_map()
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        printf("Locked mutex %p and printing thread stream map at mmr %p\n", _mutex, this);
        for (auto it = _thread_stream_map.begin(); it != _thread_stream_map.end(); ++it)
        {
            std::cout << "Thread ID: " << (void*)it->first << ", Stream: " << it->second << std::endl;
        }
    }

    void StreamedDeviceMemory::insert_thread_stream_mapping(pthread_t thread_key, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        auto it = _thread_stream_map.find(thread_key);
        if (it != _thread_stream_map.end()) _thread_stream_map.erase(it);
        _thread_stream_map.emplace(thread_key, stream);
    }

    StreamedMemory::StreamedMemory(std::shared_ptr<std::mutex> mutex, std::shared_ptr<rmm::mr::cuda_async_memory_resource> dmmr) 
        : _mutex(mutex), _dmmr(dmmr) {}

    void* StreamedMemory::allocate(size_t size, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        void* ptr = _dmmr->allocate(size, rmm::cuda_stream_view(stream));
        return ptr;
    }

    void StreamedDeviceMemory::allocate_n(void* ptrs[], const size_t sizes[], size_t n)
    {
        if (n == 0) return;

        std::lock_guard<std::mutex> guard(*_mutex);  // Lock once for the entire batch

        pthread_t thread_key = pthread_self();
        auto it = _thread_stream_map.find(thread_key);
        assert(it != _thread_stream_map.end());
        auto stream = rmm::cuda_stream_view(it->second);

        for (size_t i = 0; i < n; ++i) {
            if (sizes[i] == 0) {
                ptrs[i] = nullptr;
            } else {
                ptrs[i] = _dmmr->allocate(sizes[i], stream);
            }
        }
    }

    void StreamedMemory::deallocate(void* ptr, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        _dmmr->deallocate(ptr, 0, rmm::cuda_stream_view(stream));
    }

    void StreamedDeviceMemory::deallocate_n(void* ptrs[], size_t n)
    {
        if (n == 0) return;

        std::lock_guard<std::mutex> guard(*_mutex);  // Lock once for the entire batch

        pthread_t thread_key = pthread_self();
        auto it = _thread_stream_map.find(thread_key);
        // assert(it != _thread_stream_map.end());
        if (it == _thread_stream_map.end())
        {
            for (size_t i = 0; i < n; ++i) {
                if (ptrs[i] != nullptr) {
                    _dmmr->deallocate(ptrs[i], 0, rmm::cuda_stream_view(cudaStream_t(0)));
                }
            }
        }
        else
        {
            for (size_t i = 0; i < n; ++i) {
                if (ptrs[i] != nullptr) {
                    _dmmr->deallocate(ptrs[i], 0, rmm::cuda_stream_view(it->second));
                }
            }
        }
    }

    PinnedMemory::PinnedMemory(block_t* head, size_t alignment, std::mutex* mutex) : _head(head), _alignment(alignment), _mutex(mutex)
    {
        if (_head == nullptr || _mutex == nullptr)
        {
            throw std::runtime_error("Invalid arguments passed to PinnedMemory constructor.");
        }
    }

    size_t PinnedMemory::prefixsum_aligned_n_chunks(const size_t sizes[], size_t n, size_t prefix[])
{
    if (n == 0) return 0;

    const size_t alignment = 256; // Default alignment for pinned memory
    prefix[0] = 0;
    for (size_t i = 0; i < n - 1; ++i) {
        prefix[i + 1] = prefix[i] + IROUNDUP(sizes[i], alignment);
    }

    return prefix[n - 1] + sizes[n - 1];
}

    void* PinnedMemory::allocate(size_t size)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        return SMAQ_ALLOCATE(_head, size);
    }

    void* PinnedMemory::safe_allocate(size_t size)
    {
        while (true)
        {
            try
            {
                std::lock_guard<std::mutex> guard(*_mutex);
                return SMAQ_ALLOCATE(_head, size);
            }
            catch (std::bad_alloc const& ex)
            {
                // Introduce a small delay before retrying
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    void PinnedMemory::allocate_n(void* ptrs[], const size_t sizes[], size_t n)
    {
        if (n == 0) return;

        std::lock_guard<std::mutex> guard(*_mutex);  // Lock once for the entire batch

        for (size_t i = 0; i < n; ++i) {
            if (sizes[i] == 0) {
                ptrs[i] = nullptr;
            } else {
                ptrs[i] = SMAQ_ALLOCATE(_head, sizes[i]);
            }
        }
    }

    void PinnedMemory::deallocate_n(void* ptrs[], size_t n)
    {
        if (n == 0) return;

        std::lock_guard<std::mutex> guard(*_mutex);  // Lock once for the entire batch

        for (size_t i = 0; i < n; ++i) {
            if (ptrs[i] != nullptr) {
                SMAQ_DEALLOCATE(_head, ptrs[i]);
            }
        }
    }

    size_t PinnedMemory::allocate_n_chunks(void* chunks[], const size_t sizes[], size_t n)
    {
        if (n == 0) return 0;

        size_t total_size = 0;
        for (size_t i = 0; i < n; ++i) {
            total_size += IROUNDUP(sizes[i], _alignment);
        }

        void* base_ptr = allocate(total_size);

        size_t offset = 0;
        for (size_t i = 0; i < n; ++i) {
            chunks[i] = (sizes[i] != 0) ? static_cast<char*>(base_ptr) + offset : nullptr;
            offset += IROUNDUP(sizes[i], _alignment);
        }

        // Ensure the first chunk points to the base allocation
        chunks[0] = base_ptr;

        return total_size;
    }

    size_t PinnedMemory::allocate_n_chunks(void* chunks[], const size_t sizes[], size_t n,  size_t prefixsum[])
    {
        prefixsum[0] = 0;
        for (size_t i = 0; i < n - 1; ++i) {
            prefixsum[i + 1] = prefixsum[i] + IROUNDUP(sizes[i], _alignment);
        }

        size_t sum = prefixsum[n - 1] + sizes[n - 1];
        void* ptr = allocate(sum);

        for (size_t i = 0; i < n; ++i) {
            chunks[i] = (sizes[i] != 0) ? ((char*)ptr + prefixsum[i]) : NULL;
        }

        // 0th chunk to point to the allocated large buffer, and be non-NULL
        chunks[0] = ptr;
        return sum;
    }

    void PinnedMemory::deallocate(void* ptr)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        SMAQ_DEALLOCATE(_head, ptr);
    }

    void PinnedMemory::printBlocks()
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        printf("Locked mutex %p and printing blocks at mmr %p\n", _mutex, this);
        block_t* current = _head;
        while (current != nullptr)
        {
            printf("Block: size=%zu, free=%d, next=%p\n", current->size, current->free, (void*)current->next);
            current = current->next;
        }
    }

    PinnedHostMemory::PinnedHostMemory(std::shared_ptr<std::mutex> mutex, block_t* head, size_t alignment) : _head(head), _alignment(alignment), _mutex(mutex)
    {
        if (_head == nullptr || _mutex == nullptr)
        {
            throw std::runtime_error("Invalid arguments passed to PinnedHostMemory constructor.");
        }
    }

    void* PinnedHostMemory::allocate(size_t size)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        return SMAQ_ALLOCATE(_head, size);
    }

    void PinnedHostMemory::deallocate(void* ptr)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        SMAQ_DEALLOCATE(_head, ptr);
    }

    ManagedMemory::ManagedMemory(block_t* head, size_t alignment, std::mutex* mutex) : _head(head), _alignment(alignment), _mutex(mutex)
    {
        if (_head == nullptr || _mutex == nullptr)
        {
            throw std::runtime_error("Invalid arguments passed to ManagedMemory constructor.");
        }
    }

    void* ManagedMemory::allocate(size_t size)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        return SMAQ_ALLOCATE(_head, size);
    }

    void* ManagedMemory::safe_allocate(size_t size)
    {
        while (true)
        {
            try
            {
                std::lock_guard<std::mutex> guard(*_mutex);
                return SMAQ_ALLOCATE(_head, size);
            }
            catch (std::bad_alloc const& ex)
            {
                // Introduce a small delay before retrying
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    void ManagedMemory::deallocate(void* ptr)
    {
        std::lock_guard<std::mutex> guard(*_mutex);
        SMAQ_DEALLOCATE(_head, ptr);
    }

}} // namespace gmp::resources