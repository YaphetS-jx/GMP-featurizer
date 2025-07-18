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
        PinnedMemoryPool(size_t initial_size);
        ~PinnedMemoryPool();

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
        PinnedMemoryPool(const PinnedMemoryPool&) = delete;
        PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;

        block_t* _head;
        size_t _size;

        block_t* initialize_memory(void* ptr, size_t size);

        void* allocate_memory_first_fit(size_t size, size_t alignment = 256);

        void *allocate_memory_best_fit(size_t size, size_t alignment = 256);

        void deallocate_memory(void* ptr);
    };
}}