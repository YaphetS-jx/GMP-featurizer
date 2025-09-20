#pragma once
#ifdef GMP_USE_CUDA
#include <cuda_runtime.h>
#endif // GMP_USE_CUDA

namespace gmp { namespace group_add {

#ifndef __CUDA_ARCH__
    template <typename T>
    struct HashTable {};

#else 
    template <typename T>
    struct HashTable {
        int* keys;
        T* vals;
        int size;
        
        __device__ 
        HashTable(int* k, T* v, int s) : keys(k), vals(v), size(s) {}
        
        __device__ 
        void initialize(int local_tid, int block_size) {
            for (int i = local_tid; i < size; i += block_size) {
                keys[i] = -1;  // -1 indicates empty slot
                vals[i] = 0.0;
            }
        }
        
        __device__ 
        void flush(int local_tid, int block_size, T* global_out) {
            for (int i = local_tid; i < size; i += block_size) {
                if (keys[i] != -1 && vals[i] != 0.0) {
                    atomicAdd(global_out + keys[i], vals[i]);
                }
            }
        }
    };

    template <typename T>
    __device__ 
    void add_to_hash_table(T* __restrict__ global_out, const int key, 
        const T value, const HashTable<T>* hash_table, const int local_tid) 
    {
        // Better hash function: use key and local_tid for better distribution
        int hash_idx = (key * 31 + local_tid * 7) % hash_table->size;
        bool inserted = false;

        for (int probe = 0; probe < 16 && !inserted; ++probe) {  // Reduced probing limit
            // Quadratic probing to reduce clustering
            int offset = probe * probe + local_tid;
            int slot = (hash_idx + offset) % hash_table->size;

            int old_key = atomicCAS(&hash_table->keys[slot], -1, key);

            if (old_key == -1) {
                // We claimed an empty slot
                atomicAdd(&hash_table->vals[slot], value);
                inserted = true;
            } else if (old_key == key) {
                // Slot already contains our key
                atomicAdd(&hash_table->vals[slot], value);
                inserted = true;
            }
        }

        // Fallback: if hash table is full, use global atomic directly
        if (!inserted) {
            atomicAdd(global_out + key, value);
        }
    }
#endif

}} // namespace gmp::group_add