#include "memory_manager.hpp"
#include <stdint.h>
#include <assert.h>

namespace gmp { namespace resources {

    block_t* initialize_memory(void* ptr, size_t size)
    {
        assert(size >= sizeof(block_t));
        block_t* head = (block_t *)ptr;
        head->size = size - sizeof(block_t);
        head->next = NULL;
        head->free = 1;
        return head;
    }

    void *allocate_memory_best_fit(block_t* head, size_t size, size_t alignment)
    {
        block_t *current = head;
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
            new_block->free = 1;

            best_fit->size = size + offset;
            best_fit->next = new_block;
        }
        best_fit->free = 0;

        // Store the original pointer just before the aligned memory
        void **original_ptr = (void **)((char *)(best_fit + 1) + offset - sizeof(void *));
        *original_ptr = (void *)best_fit;

        return (void *)((char *)(best_fit + 1) + offset);
    }

    void *allocate_memory_first_fit(block_t* head, size_t size, size_t alignment)
    {
        block_t *current = head;
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
                        new_block->free = 1;

                        current->size = size + offset;
                        current->next = new_block;
                    }
                    current->free = 0;

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

    void deallocate_memory(block_t* head, void *ptr)
    {
        if (ptr == NULL) return;

        // Retrieve the original pointer stored just before the aligned memory
        void **original_ptr = (void **)((char *)ptr - sizeof(void *));
        block_t *block_to_free = (block_t *)(*original_ptr);

        block_to_free->free = 1;

        block_t *current = head;
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

}} // namespace gmp::resources