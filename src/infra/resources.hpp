#pragma once
#include <memory>
#include <sys/sysinfo.h>
#include <iostream>
#include <iomanip>
#include <thread>
#include "error.hpp"
#include "thread_pool.hpp"
#ifdef GMP_ENABLE_CUDA
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#endif

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

        #ifdef GMP_ENABLE_CUDA
        // get gpu device memory pool
        rmm::mr::cuda_async_memory_resource* get_gpu_device_memory_pool() const {
            if (!gpu_device_memory_pool_) {
                auto [free_memory, total_memory] = rmm::available_device_memory();
                // Use a reasonable initial pool size (1GB) instead of all available memory
                size_t initial_pool_size = std::min(free_memory / 4, size_t(1 << 30)); // 1GB or 25% of free memory, whichever is smaller
                gpu_device_memory_pool_ = std::make_unique<rmm::mr::cuda_async_memory_resource>(initial_pool_size);
            }
            return gpu_device_memory_pool_.get();
        }

        // get pinned host memory pool
        rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>* get_pinned_host_memory_pool() const {
            if (!upstream_resource_) {
                upstream_resource_ = std::make_unique<rmm::mr::pinned_host_memory_resource>();
            }
            if (!pinned_pool_) {
                auto [free_memory, total_memory] = rmm::available_device_memory();
                pinned_pool_ = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>>(
                    *upstream_resource_,
                    1 << 27, // 128MB initial pool size
                    total_memory // maximum pool size
                );
            }
            return pinned_pool_.get();
        }
        #endif

    private:
        mutable std::unique_ptr<ThreadPool> thread_pool_;
        mutable std::mutex pool_mutex_;
        #ifdef GMP_ENABLE_CUDA        
        mutable std::unique_ptr<rmm::mr::cuda_async_memory_resource> gpu_device_memory_pool_;
        mutable std::unique_ptr<rmm::mr::pinned_host_memory_resource> upstream_resource_;
        mutable std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>> pinned_pool_;
        #endif

        gmp_resource() = default;
        ~gmp_resource() = default;
        gmp_resource(const gmp_resource&) = delete;
        gmp_resource& operator=(const gmp_resource&) = delete;
    };
}}

