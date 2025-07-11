#pragma once
#include <stddef.h>

namespace gmp { namespace resources {

    // memory alignment
    #ifndef MEMORY_ALIGNMENT
    #define MEMORY_ALIGNMENT 256
    #endif

    #ifndef SMAQ_ALLOCATE
    #define SMAQ_ALLOCATE(header, size) allocate_memory_best_fit((header), (size), MEMORY_ALIGNMENT)
    #endif

    #ifndef SMAQ_DEALLOCATE
    #define SMAQ_DEALLOCATE(header, ptr) deallocate_memory((header), (ptr))
    #endif

    typedef struct block
    {
        size_t size;
        struct block *next;
        int free;
    } block_t;

    block_t* initialize_memory(void* ptr, size_t size);
    void* allocate_memory_first_fit(block_t* head, size_t size, size_t alignment);
    void *allocate_memory_best_fit(block_t* head, size_t size, size_t alignment);
    void deallocate_memory(block_t* head, void* ptr);

}} // namespace gmp::resources