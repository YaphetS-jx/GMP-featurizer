#pragma once
#include <cstdint>

namespace gmp { namespace resources {
    
    typedef struct block
    {
    size_t size;
    struct block *next;
    bool free;
    } block_t;

    class PinnedMemoryPool
    {
    public:
        static PinnedMemoryPool& instance(size_t initial_size = uint64_t(1) << 30);
        void* allocate(size_t size);
        void deallocate(void* ptr);
        
        // Pool information methods
        size_t get_total_size() const;
        size_t get_used_size() const;
        size_t get_free_size() const;
        size_t get_block_count() const;
        size_t get_free_block_count() const;
        double get_utilization_percentage() const;
        
        // Debug information methods
        void print_detailed_blocks() const;

    private:
        PinnedMemoryPool(size_t initial_size);
        ~PinnedMemoryPool();

        PinnedMemoryPool(const PinnedMemoryPool&) = delete;
        PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

        block_t* _head;
        size_t _size;

        block_t* initialize_memory(void* ptr, size_t size);

        void* allocate_memory_first_fit(size_t size, size_t alignment = 256);

        void *allocate_memory_best_fit(size_t size, size_t alignment = 256);

        void deallocate_memory(void* ptr);
    };

    // Custom allocator that wraps pinned_host_memory_resource
    template <typename T>
    class pinned_host_allocator {
    public:
        using value_type = T;
        pinned_host_allocator() noexcept = default;

        // Allow rebinding to other types
        template <typename U>
        pinned_host_allocator(const pinned_host_allocator<U>&) noexcept {}

        T* allocate(size_t n) {
            if (n == 0) return nullptr;
            return static_cast<T*>(PinnedMemoryPool::instance().allocate(n * sizeof(T)));
        }

        void deallocate(T* p, size_t n) noexcept {
            if (p != nullptr) {
                PinnedMemoryPool::instance().deallocate(p);
            }
        }

        // All instances of this allocator are interchangeable:
        bool operator==(const pinned_host_allocator&) const noexcept { return true; }
        bool operator!=(const pinned_host_allocator& a) const noexcept { return !(*this == a); }
    };
}}