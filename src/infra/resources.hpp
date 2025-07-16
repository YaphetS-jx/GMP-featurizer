#pragma once
#include <memory>
#include <sys/sysinfo.h>
#include <iostream>
#include <iomanip>
#include <thread>
#include "error.hpp"
#include "thread_pool.hpp"
#include "pinned_memory_pool.hpp"

namespace gmp { namespace resources {

    // collection of all resources
    class gmp_resource {
    public:
        // Thread-safe singleton instance
        static gmp_resource& instance() {
            static gmp_resource instance;
            return instance;
        }

        // get thread pool (lazy initialization)
        ThreadPool& get_thread_pool(const int num_threads) const {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            
            // Create thread pool only when first accessed
            if (!thread_pool_) {
                thread_pool_ = std::make_unique<ThreadPool>(num_threads);
                std::cout << "Thread pool created on with " 
                          << thread_pool_->get_thread_count() << " threads" << std::endl;
            }
            
            return *thread_pool_;
        }

        // get host memory pool
        PinnedMemoryPool& get_pinned_memory_pool() const {
            return PinnedMemoryPool::instance();
        }

    private:
        mutable std::unique_ptr<ThreadPool> thread_pool_;
        mutable std::mutex pool_mutex_;

        gmp_resource() = default;
        ~gmp_resource() = default;
        gmp_resource(const gmp_resource&) = delete;
        gmp_resource& operator=(const gmp_resource&) = delete;
    };
}}

