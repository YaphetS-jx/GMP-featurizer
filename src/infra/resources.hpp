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

    #ifdef GMP_ENABLE_CUDA
    // foward declare the resource classes
    template <typename T> class pinned_host_allocator;
    class pinned_memory_manager;
    class device_memory_manager;
    #endif

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

        cudaStream_t get_stream() const {
            if (!stream_) {
                cudaError_t status = cudaStreamCreate(&stream_);
                if (status != cudaSuccess) {
                    throw std::runtime_error("Failed to create CUDA stream");
                }
            }
            return stream_;
        }

        // get device memory manager
        device_memory_manager* get_device_memory_manager() const {
            if (!device_memory_manager_) {
                device_memory_manager_ = std::make_unique<device_memory_manager>();
            }
            return device_memory_manager_.get();
        }

        // get pinned memory manager
        pinned_memory_manager* get_pinned_memory_manager() const {
            if (!pinned_memory_manager_) {
                pinned_memory_manager_ = std::make_unique<pinned_memory_manager>();
            }
            return pinned_memory_manager_.get();
        }
        
        #endif

        // Explicit cleanup function to be called before program exit
        void cleanup() {
            #ifdef GMP_ENABLE_CUDA
            if (stream_) {
                cudaStreamSynchronize(stream_);
                cudaStreamDestroy(stream_);
                stream_ = nullptr;
            }
            
            // Explicitly clean up RMM resources
            pinned_pool_.reset();
            upstream_resource_.reset();
            gpu_device_memory_pool_.reset();
            #endif
        }

    private:
        mutable std::unique_ptr<ThreadPool> thread_pool_;
        mutable std::mutex pool_mutex_;
        #ifdef GMP_ENABLE_CUDA        
        mutable std::unique_ptr<rmm::mr::cuda_async_memory_resource> gpu_device_memory_pool_;
        mutable std::unique_ptr<rmm::mr::pinned_host_memory_resource> upstream_resource_;
        mutable std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>> pinned_pool_;
        mutable std::unique_ptr<device_memory_manager> device_memory_manager_;
        mutable std::unique_ptr<pinned_memory_manager> pinned_memory_manager_;
        mutable cudaStream_t stream_;
        #endif

        gmp_resource() = default;
        ~gmp_resource() {
            cleanup();
        }
        gmp_resource(const gmp_resource&) = delete;
        gmp_resource& operator=(const gmp_resource&) = delete;
    };

    #ifdef GMP_ENABLE_CUDA
    // Custom allocator that wraps pinned_host_memory_resource
    template <typename T>
    class pinned_host_allocator {
    public:
        using value_type = T;
        pinned_host_allocator() noexcept 
            : _pool(gmp::resources::gmp_resource::instance().get_pinned_host_memory_pool()) 
        {}

        // Allow rebinding to other types
        template <typename U>
        pinned_host_allocator(const pinned_host_allocator<U>&) noexcept {}

        T* allocate(size_t n) {
            if (n == 0) return nullptr;
            return static_cast<T*>(_pool->allocate(n * sizeof(T)));
        }

        void deallocate(T* p, size_t n) noexcept {
            if (p != nullptr) {
                _pool->deallocate(p, n);
            }
        }

        // All instances of this allocator are interchangeable:
        bool operator==(const pinned_host_allocator&) const noexcept { return true; }
        bool operator!=(const pinned_host_allocator& a) const noexcept { return !(*this == a); }
    private:
        rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>* _pool;
    };

    class pinned_memory_manager {
    public:
        pinned_memory_manager() noexcept 
            : _pool(gmp::resources::gmp_resource::instance().get_pinned_host_memory_pool()) 
        {}

        void* allocate(size_t n) {
            if (n == 0) return nullptr;
            return _pool->allocate(n);
        }

        void deallocate(void* p) noexcept {
            if (p != nullptr) {
                _pool->deallocate(p, 0);
            }
        }
    private:
        rmm::mr::pool_memory_resource<rmm::mr::pinned_host_memory_resource>* _pool;
    };

    class device_memory_manager {
    public:
        device_memory_manager() noexcept 
            : _pool(gmp::resources::gmp_resource::instance().get_gpu_device_memory_pool()) 
        {}

        void* allocate(size_t n, cudaStream_t stream) {
            if (n == 0) return nullptr;
            return _pool->allocate(n, rmm::cuda_stream_view(stream));
        }

        void deallocate(void* p, cudaStream_t stream) noexcept {
            if (p != nullptr) {
                _pool->deallocate(p, 0, rmm::cuda_stream_view(stream));
            }
        }
    private: 
        rmm::mr::cuda_async_memory_resource* _pool;
    };
    #endif
}}

