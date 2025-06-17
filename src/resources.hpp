#pragma once
#include <memory>
#include <sys/sysinfo.h>
#include <iostream>
#include <iomanip>
#include "error.hpp"

namespace gmp { namespace resources {

    template <typename T, typename PoolType>
    class pool_allocator;

    // Host memory pool
    template <typename PoolType>
    class HostMemory {
    public:
        // Thread-safe singleton instance
        static HostMemory& instance() {
            static HostMemory instance;
            return instance;
        }

        // initialize host memory pool
        void initialize() {}

        ~HostMemory() = default;

        // get memory pool allocator
        template <typename T>
        pool_allocator<T, PoolType> get_allocator() const {}

        // get memory pool
        PoolType& get_pool() const {}

        // print memory info
        void print_memory_info() const {}
        
    private: 
        HostMemory() = default;
        HostMemory(const HostMemory&) = delete;
        HostMemory& operator=(const HostMemory&) = delete;
    };

    // collection of all resources
    template <typename PoolType>
    class gmp_resource {
    public:
        // Thread-safe singleton instance
        static gmp_resource& instance() {
            static gmp_resource instance;
            return instance;
        }

        // get host memory pool
        HostMemory<PoolType>& get_host_memory() const {
            return HostMemory<PoolType>::instance();
        }

    private:
        gmp_resource() {}
        ~gmp_resource() = default;
        gmp_resource(const gmp_resource&) = delete;
        gmp_resource& operator=(const gmp_resource&) = delete;
    };

    // Pool allocator
    template <typename T, typename PoolType> 
    class pool_allocator {
    private:
        PoolType* pool_;
        
    public:
        using value_type = T;
        
        // Default constructor (needed for std::vector)
        pool_allocator(PoolType& pool) : pool_(&pool) {
            assert(pool_);
            assert(pool_size() >= sizeof(T));
        }
        
        template <typename U>
        pool_allocator(pool_allocator<U, PoolType> const& other) : pool_(other.get_pool()) {
            assert(pool_ && pool_size() >= sizeof(T));
        }
        
        // Get the pool pointer
        PoolType* get_pool() const { return pool_; }
        
        // allocator
        T *allocate(const size_t n) {
            if (!pool_) {
                GMP_EXIT(error_t::memory_bad_alloc);                
            }
            T* ret = static_cast<T*>(pool_->ordered_malloc(n));
            if (!ret && n) {
                GMP_EXIT(error_t::memory_bad_alloc);
            }
            return ret;
        }
        
        // deallocator
        void deallocate(T* ptr, const size_t n) {
            if (pool_ && ptr && n) pool_->ordered_free(ptr, n);
        }

        // construct element
        template<typename U, typename... Args>
        void construct(U* p, Args&&... args) {
            ::new((void*)p) U(std::forward<Args>(args)...);
        }

        // destroy element
        template<typename U>
        void destroy(U* p) {
            p->~U();
        }
        
        // pool size
        size_t pool_size() const { 
            return pool_ ? pool_->get_requested_size() : 0; 
        }

        // equality operators
        bool operator==(const pool_allocator& other) const {
            return pool_ == other.get_pool();
        }
        
        bool operator!=(const pool_allocator& other) const {
            return !(*this == other);
        }
    };
}}

