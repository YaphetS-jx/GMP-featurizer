#pragma once

#include <boost/pool/pool.hpp>
#include <memory>
#include <iostream>
#include <vector>
#include "error.hpp"

namespace gmp { namespace resources {

    using PoolType = boost::pool<boost::default_user_allocator_malloc_free>;

    // Pool-aware deleter
    template <typename T, typename PoolType>
    struct pool_deleter {
        PoolType* pool;
        size_t n;
        void operator()(T* ptr) const {
            if (ptr) pool->ordered_free(ptr, n);
            // std::cout << "deleter deallocated " << n << " blocks" << std::endl;
        }
    };

    template <typename T, typename PoolType>
    class pool_unique_ptr {
        using Deleter = pool_deleter<T, PoolType>;
        std::unique_ptr<T, Deleter> ptr_;
    public:
        pool_unique_ptr(T* ptr, PoolType* pool, size_t n = 1)
            : ptr_(ptr, Deleter{pool, n}) {}
        T* get() const { return ptr_.get(); }
        T& operator*() const { return *ptr_; }
        T* operator->() const { return ptr_.get(); }
        explicit operator bool() const { return static_cast<bool>(ptr_); }
        
        // Add methods needed for conversion
        T* release() { return ptr_.release(); }
        PoolType* get_pool() const { return ptr_.get_deleter().pool; }
        size_t get_count() const { return ptr_.get_deleter().n; }
        
        // Move only
        pool_unique_ptr(pool_unique_ptr&&) = default;
        pool_unique_ptr& operator=(pool_unique_ptr&&) = default;
        pool_unique_ptr(const pool_unique_ptr&) = delete;
        pool_unique_ptr& operator=(const pool_unique_ptr&) = delete;
    };

    // Pool-aware make_unique (like std::make_unique, but uses the pool)
    template <typename T, typename PoolType, typename... Args>
    pool_unique_ptr<T, PoolType> make_pool_unique(PoolType& pool, Args&&... args) {
        void* mem = pool.ordered_malloc(1);
        if (!mem) {
            GMP_EXIT(error_t::memory_bad_alloc);
        }
        T* obj = new (mem) T(std::forward<Args>(args)...); // placement new
        return pool_unique_ptr<T, PoolType>(obj, &pool, 1);
    }

    // Pool-aware make_shared (like std::make_shared, but uses the pool)
    template <typename T, typename PoolType, typename... Args>
    std::shared_ptr<T> make_pool_shared(PoolType& pool, Args&&... args) {
        void* mem = pool.ordered_malloc(1);
        if (!mem) {
            GMP_EXIT(error_t::memory_bad_alloc);
        }
        T* obj = new (mem) T(std::forward<Args>(args)...); // placement new
        return std::shared_ptr<T>(obj, [&pool](T* p) {
            if (p) {
                p->~T(); // call destructor
                pool.ordered_free(p, 1);
                // std::cout << "shared_ptr deallocated 1 blocks" << std::endl;
            }
        });
    }

    // Pool allocator     
    template <typename T> struct pool_allocator {
        using value_type = T;
        
        // Default constructor (needed for std::vector)
        pool_allocator() : _pool(nullptr) {}
        
        pool_allocator(PoolType& pool) : _pool(&pool) {
            assert(pool_size() >= sizeof(T));
        }
        template <typename U>
        pool_allocator(pool_allocator<U> const& other) : _pool(other._pool) {
            assert(_pool && pool_size() >= sizeof(T));
        }
        // allocator
        T *allocate(const size_t n) {
            if (!_pool) {
                GMP_EXIT(error_t::memory_bad_alloc);                
            }
            T* ret = static_cast<T*>(_pool->ordered_malloc(n));
            if (!ret && n) {
                GMP_EXIT(error_t::memory_bad_alloc);
            }
            return ret;
        }
        // deallocator
        void deallocate(T* ptr, const size_t n) {
            if (_pool && ptr && n) _pool->ordered_free(ptr, n);
        }
        // pool size
        size_t pool_size() const { 
            return _pool ? _pool->get_requested_size() : 0; 
        }

        // equality operators
        bool operator==(const pool_allocator& other) const {
            return _pool == other._pool;
        }
        bool operator!=(const pool_allocator& other) const {
            return !(*this == other);
        }

        // Add propagate_on_container_move_assignment
        using propagate_on_container_move_assignment = std::true_type;
        using propagate_on_container_copy_assignment = std::true_type;
        using propagate_on_container_swap = std::true_type;
        using is_always_equal = std::false_type;
    private:
        PoolType* _pool;
    };

    // print memory info
    inline void print_boost_pool_memory_info(PoolType& pool) {
        auto format_size = [](uint64_t bytes) -> std::string {
            const char* units[] = {"B", "KB", "MB", "GB", "TB"};
            int unit_index = 0;
            double size = static_cast<double>(bytes);
            
            while (size >= 1024.0 && unit_index < 4) {
                size /= 1024.0;
                unit_index++;
            }

            // Check if we can represent this in the next unit up
            if (unit_index < 4 && size >= 1000.0) {
                size /= 1024.0;
                unit_index++;
            }
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
            return ss.str();
        };

        std::cout << "Host memory pool info:" << std::endl;
        std::cout << "Node size: " << format_size(pool.get_requested_size()) << std::endl;
        std::cout << "Next size: " << format_size(pool.get_next_size()) << std::endl;
        std::cout << "Max size: " << format_size(pool.get_max_size()) << std::endl; 
    }
}}
