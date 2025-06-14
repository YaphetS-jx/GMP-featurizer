#pragma once

#include <memory>
#include <sys/sysinfo.h>
#include <iostream>
#include <iomanip>
#include "boost_pool.hpp"

namespace gmp { namespace resources {

    template <typename T>
    class pool_allocator;

    // Host memory pool
    class HostMemory {
    public:
        // Thread-safe singleton instance
        static HostMemory& instance() {
            static HostMemory instance;
            return instance;
        }

        // initialize host memory pool
        void initialize(size_t node_size = DEFAULT_NODE_SIZE, size_t next_size = DEFAULT_NEXT_SIZE, size_t max_size = DEFAULT_MAX_SIZE) {
            pool_ = std::make_unique<PoolType>(node_size, next_size, max_size);
        }

        ~HostMemory() = default;

        // get memory pool allocator
        template <typename T>
        pool_allocator<T> get_allocator() const {
            return pool_allocator<T>(*pool_);
        }

        // get memory pool
        PoolType& get_pool() const {
            return *pool_;
        }

        // print memory info
        void print_memory_info() const {
            print_boost_pool_memory_info(*pool_);
        }

        constexpr static size_t DEFAULT_NODE_SIZE = (1<<6);         // default for 64 bytes memory for node size
        constexpr static size_t DEFAULT_NEXT_SIZE = (1<<26);    // default for 64MB memory for next block size 
        constexpr static size_t DEFAULT_MAX_SIZE = (1<<30);     // default for 1GB memory for max block size 
        
    private: 
        HostMemory() = default;
        HostMemory(const HostMemory&) = delete;
        HostMemory& operator=(const HostMemory&) = delete;

        std::unique_ptr<PoolType> pool_;
    };

    // collection of all resources
    class gmp_resource {
    public:
        // Thread-safe singleton instance
        static gmp_resource& instance(size_t node_size = HostMemory::DEFAULT_NODE_SIZE, size_t next_size = HostMemory::DEFAULT_NEXT_SIZE, size_t max_size = HostMemory::DEFAULT_MAX_SIZE) {
            static gmp_resource instance(node_size, next_size, max_size);
            return instance;
        }

        // get host memory pool
        HostMemory& get_host_memory() const {
            return HostMemory::instance();
        }

    private:
        gmp_resource(size_t node_size = HostMemory::DEFAULT_NODE_SIZE, size_t next_size = HostMemory::DEFAULT_NEXT_SIZE, size_t max_size = HostMemory::DEFAULT_MAX_SIZE) {
            HostMemory::instance().initialize(node_size, next_size, max_size);
        }
        ~gmp_resource() = default;        
        gmp_resource(const gmp_resource&) = delete;
        gmp_resource& operator=(const gmp_resource&) = delete;
    };

    inline PoolType& get_default_pool() {
        return gmp_resource::instance().get_host_memory().get_pool();
    }

    // Pool allocator
    template <typename T> struct pool_allocator {
    private:
        PoolType* pool_;
        
    public:
        using value_type = T;
        
        // Default constructor (needed for std::vector)
        pool_allocator() : pool_(&(get_default_pool())) {}
        
        pool_allocator(PoolType& pool) : pool_(&pool) {
            assert(pool_size() >= sizeof(T));
            assert(pool_);
        }
        
        template <typename U>
        pool_allocator(pool_allocator<U> const& other) : pool_(other.get_pool()) {
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

        // Add propagate_on_container_move_assignment
        using propagate_on_container_move_assignment = std::true_type;
        using propagate_on_container_copy_assignment = std::true_type;
        using propagate_on_container_swap = std::true_type;
        using is_always_equal = std::false_type;
    };
}}

