#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "pinned_memory_pool.hpp"
#include "util.hpp"

namespace gmp { namespace resources {

    PinnedMemoryPool::PinnedMemoryPool(size_t initial_size)
    {
        void* ptr = nullptr;
        cudaError_t err = cudaMallocHost(&ptr, initial_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned memory");
        }
        _head = initialize_memory(ptr, initial_size);
        _size = initial_size;
    }

    PinnedMemoryPool::~PinnedMemoryPool()
    {
        cudaError_t err = cudaFreeHost(_head);
        if (err != cudaSuccess) {
            // In destructors, we should not throw exceptions
            // Just log the error or handle it gracefully
            fprintf(stderr, "Warning: Failed to free pinned memory: %s\n", cudaGetErrorString(err));
        }
    }

    void* PinnedMemoryPool::allocate(size_t size)
    {
        return allocate_memory_best_fit(size);
    }

    void PinnedMemoryPool::deallocate(void* ptr)
    {
        deallocate_memory(ptr);
    }

    // Pool information methods
    size_t PinnedMemoryPool::get_total_size() const
    {
        return _size;
    }

    size_t PinnedMemoryPool::get_used_size() const
    {
        size_t used_size = 0;
        block_t* current = _head;
        
        while (current != nullptr) {
        if (!current->free) {
            used_size += current->size;
        }
        current = current->next;
        }
        
        return used_size;
    }

    size_t PinnedMemoryPool::get_free_size() const
    {
        return get_total_size() - get_used_size();
    }

    size_t PinnedMemoryPool::get_block_count() const
    {
        size_t count = 0;
        block_t* current = _head;
        
        while (current != nullptr) {
        count++;
        current = current->next;
        }
        
        return count;
    }

    size_t PinnedMemoryPool::get_free_block_count() const
    {
        size_t count = 0;
        block_t* current = _head;
        
        while (current != nullptr) {
        if (current->free) {
            count++;
        }
        current = current->next;
        }
        
        return count;
    }

    double PinnedMemoryPool::get_utilization_percentage() const
    {
        if (_size == 0) return 0.0;
        return (static_cast<double>(get_used_size()) / static_cast<double>(_size)) * 100.0;
    }

    void PinnedMemoryPool::print_detailed_blocks() const
    {
        std::cout << "\n--- Detailed Block Information ---" << std::endl;
        block_t* current = _head;
        int block_num = 0;
        
        while (current != nullptr) {
        std::cout << "Block " << block_num << ": ";
        std::cout << "Size=" << gmp::util::format_bytes(current->size) << ", ";
        std::cout << "Status=" << (current->free ? "FREE" : "USED") << ", ";
        std::cout << "Address=" << std::hex << (void*)current << std::dec;
        
        if (current->next != nullptr) {
            std::cout << ", Next=" << std::hex << (void*)current->next << std::dec;
        } else {
            std::cout << ", Next=NULL";
        }
        std::cout << std::endl;
        
        current = current->next;
        block_num++;
        }
    }

    block_t* PinnedMemoryPool::initialize_memory(void* ptr, size_t size)
    {
        assert(size >= sizeof(block_t));
        block_t* head = (block_t *)ptr;
        head->size = size - sizeof(block_t);
        head->next = NULL;
        head->free = true;
        return head;
    }

    void *PinnedMemoryPool::allocate_memory_best_fit(size_t size, size_t alignment)
    {
        block_t *current = _head;
        block_t *best_fit = NULL;
        size_t best_fit_size = SIZE_MAX;

        // Find the best fit block
        while (current != NULL)
        {
            if (current->free)
            {
                size_t aligned_address = (size_t)(((uintptr_t)(current + 1) + (uintptr_t)sizeof(void*) + (alignment - 1)) & ~(alignment - 1));
                size_t offset = aligned_address - (size_t)(current + 1);

                if (current->size >= size + offset && current->size < best_fit_size)
                {
                    best_fit = current;
                    best_fit_size = current->size;
                }
            }
            current = current->next;
        }

        // If no suitable block was found, return NULL
        if (best_fit == NULL)
        {
            return NULL;
        }

        // Allocate memory from the best fit block
        size_t aligned_address = (size_t)(((uintptr_t)(best_fit + 1) + (uintptr_t)sizeof(void*) + (alignment - 1)) & ~(alignment - 1));
        size_t offset = aligned_address - (size_t)(best_fit + 1);

        if (best_fit->size >= size + offset + sizeof(block_t))
        {
            block_t *new_block = (block_t *)((char *)(best_fit + 1) + offset + size);
            new_block->size = best_fit->size - size - offset - sizeof(block_t);
            new_block->next = best_fit->next;
            new_block->free = true;

            best_fit->size = size + offset;
            best_fit->next = new_block;
        }
        best_fit->free = false;

        // Store the original pointer just before the aligned memory
        void **original_ptr = (void **)((char *)(best_fit + 1) + offset - sizeof(void *));
        *original_ptr = (void *)best_fit;

        return (void *)((char *)(best_fit + 1) + offset);
    }

    void *PinnedMemoryPool::allocate_memory_first_fit(size_t size, size_t alignment)
    {
        block_t *current = _head;
        while (current != NULL)
        {
            if (current->free)
            {
                size_t aligned_address = (size_t)(((uintptr_t)(current + 1) + (uintptr_t)sizeof(void*) + (alignment - 1)) & ~(alignment - 1));
                size_t offset = aligned_address - (size_t)(current + 1);

            if (current->size >= size + offset)
            {
                if (current->size >= size + offset + sizeof(block_t))
                {
                    block_t *new_block = (block_t *)((char *)(current + 1) + offset + size);
                    new_block->size = current->size - size - offset - sizeof(block_t);
                    new_block->next = current->next;
                    new_block->free = true;

                    current->size = size + offset;
                    current->next = new_block;
                }
                current->free = false;

                // Store the original pointer just before the aligned memory
                void **original_ptr = (void **)((char *)(current + 1) + offset - sizeof(void *));
                *original_ptr = (void *)current;

                return (void *)((char *)(current + 1) + offset);
            }
            }
            current = current->next;
        }
        return NULL;
    }

    void PinnedMemoryPool::deallocate_memory(void *ptr)
    {
        if (ptr == NULL) return;

        // Retrieve the original pointer stored just before the aligned memory
        void **original_ptr = (void **)((char *)ptr - sizeof(void *));
        block_t *block_to_free = (block_t *)(*original_ptr);

        block_to_free->free = true;

        block_t *current = _head;
        block_t *prev = NULL;

        while (current != NULL)
        {
            if (current == block_to_free)
            {
                if (prev != NULL && prev->free)
                {
                    prev->size += current->size + sizeof(block_t);
                    prev->next = current->next;
                    current = prev;
                }

                if (current->next != NULL && current->next->free)
                {
                    current->size += current->next->size + sizeof(block_t);
                    current->next = current->next->next;
                }
                return;
            }

            prev = current;
            current = current->next;
        }
    }
}}