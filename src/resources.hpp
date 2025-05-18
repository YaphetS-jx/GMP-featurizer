#pragma once

#include <memory>
#include <sys/sysinfo.h>
#include <iostream>
#include <iomanip>
#include "boost_pool.hpp"

namespace gmp { namespace resources {
    
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

        constexpr static size_t DEFAULT_NODE_SIZE = (1<<7);         // default for 128 bytes memory for node size
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

}}

