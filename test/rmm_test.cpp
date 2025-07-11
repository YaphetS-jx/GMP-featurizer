#include <iostream>
#include <vector>
#include <thread>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "rmm_pool.hpp"
#include "memory_manager.hpp"
#include <chrono>

class RMMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cudaError_t cuda_status = cudaSetDevice(0);
        ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to set CUDA device: " << cudaGetErrorString(cuda_status);
        
        // Create CUDA stream
        cuda_status = cudaStreamCreate(&stream);
        ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to create CUDA stream: " << cudaGetErrorString(cuda_status);
        
        // Initialize memory managers
        dmmr = std::make_shared<rmm::mr::cuda_async_memory_resource>();
        shared_mutex = std::make_shared<std::mutex>();
    }
    
    void TearDown() override {
        cudaStreamDestroy(stream);
    }
    
    cudaStream_t stream;
    std::shared_ptr<rmm::mr::cuda_async_memory_resource> dmmr;
    std::shared_ptr<std::mutex> shared_mutex;
};

TEST_F(RMMTest, StreamedDeviceMemoryBasicAllocation) {
    std::cout << "Testing StreamedDeviceMemory basic allocation" << std::endl;
    
    std::mutex mutex;
    gmp::resources::StreamedDeviceMemory streamed_dev_mem(&mutex, dmmr.get());
    
    // Register current thread with stream
    streamed_dev_mem.insert_thread_stream_mapping(pthread_self(), stream);
    
    // Test allocation
    size_t test_size = 1024;
    void* ptr = streamed_dev_mem.allocate(test_size);
    ASSERT_NE(ptr, nullptr) << "Allocation failed";
    std::cout << "Allocated " << test_size << " bytes at " << ptr << std::endl;
    
    // Test deallocation
    streamed_dev_mem.deallocate(ptr);
    std::cout << "Deallocated memory at " << ptr << std::endl;
}

TEST_F(RMMTest, StreamedMemoryBasicAllocation) {
    std::cout << "Testing StreamedMemory basic allocation" << std::endl;
    
    gmp::resources::StreamedMemory streamed_mem(shared_mutex, dmmr);
    
    size_t test_size = 1024;
    void* ptr = streamed_mem.allocate(test_size, stream);
    ASSERT_NE(ptr, nullptr) << "Allocation failed";
    std::cout << "Allocated " << test_size << " bytes at " << ptr << std::endl;
    
    streamed_mem.deallocate(ptr, stream);
    std::cout << "Deallocated memory at " << ptr << std::endl;
}

TEST_F(RMMTest, MultipleAllocations) {
    std::cout << "Testing multiple allocations" << std::endl;
    
    std::mutex mutex;
    gmp::resources::StreamedDeviceMemory streamed_dev_mem(&mutex, dmmr.get());
    streamed_dev_mem.insert_thread_stream_mapping(pthread_self(), stream);
    
    std::vector<size_t> sizes = {512, 1024, 2048, 4096};
    std::vector<void*> ptrs(sizes.size());
    
    size_t total_allocated = streamed_dev_mem.allocate_n_chunks(ptrs.data(), sizes.data(), sizes.size());
    ASSERT_GT(total_allocated, 0) << "Batch allocation failed";
    std::cout << "Total allocated: " << total_allocated << " bytes" << std::endl;
    
    for (size_t i = 0; i < ptrs.size(); ++i) {
        ASSERT_NE(ptrs[i], nullptr) << "Chunk " << i << " allocation failed";
        std::cout << "Chunk " << i << ": " << sizes[i] << " bytes at " << ptrs[i] << std::endl;
    }
    
    // When using allocate_n_chunks, only deallocate the base pointer (ptrs[0])
    // since all chunks are part of the same allocation
    streamed_dev_mem.deallocate(ptrs[0]);
    std::cout << "Deallocated base allocation" << std::endl;
}

TEST_F(RMMTest, SafeAllocation) {
    std::cout << "Testing safe allocation" << std::endl;
    
    std::mutex mutex;
    gmp::resources::StreamedDeviceMemory streamed_dev_mem(&mutex, dmmr.get());
    streamed_dev_mem.insert_thread_stream_mapping(pthread_self(), stream);
    
    size_t test_size = 1024;
    void* ptr = streamed_dev_mem.safe_allocate(test_size);
    ASSERT_NE(ptr, nullptr) << "Safe allocation failed";
    std::cout << "Safely allocated " << test_size << " bytes at " << ptr << std::endl;
    
    streamed_dev_mem.deallocate(ptr);
    std::cout << "Deallocated safe allocation" << std::endl;
}

TEST_F(RMMTest, PinnedMemoryAllocation) {
    std::cout << "Testing PinnedMemory allocation" << std::endl;
    
    // Allocate a large block of pinned memory for testing
    size_t pool_size = 1024 * 1024; // 1MB
    void* pool_ptr = nullptr;
    cudaError_t cuda_status = cudaMallocHost(&pool_ptr, pool_size);
    ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to allocate pinned host memory";
    
    std::mutex mutex;
    gmp::resources::block_t* head = gmp::resources::initialize_memory(pool_ptr, pool_size);
    ASSERT_NE(head, nullptr) << "Failed to initialize memory pool";
    
    gmp::resources::PinnedMemory pinned_mem(head, 256, &mutex);
    
    // Test allocation
    size_t test_size = 1024;
    void* ptr = pinned_mem.allocate(test_size);
    ASSERT_NE(ptr, nullptr) << "Pinned memory allocation failed";
    std::cout << "Allocated " << test_size << " bytes in pinned memory at " << ptr << std::endl;
    
    // Test deallocation
    pinned_mem.deallocate(ptr);
    std::cout << "Deallocated pinned memory at " << ptr << std::endl;
    
    // Cleanup
    cudaFreeHost(pool_ptr);
}

TEST_F(RMMTest, PinnedHostMemoryAllocation) {
    std::cout << "Testing PinnedHostMemory allocation" << std::endl;
    
    // Allocate a large block of pinned memory for testing
    size_t pool_size = 1024 * 1024; // 1MB
    void* pool_ptr = nullptr;
    cudaError_t cuda_status = cudaMallocHost(&pool_ptr, pool_size);
    ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to allocate pinned host memory";
    
    gmp::resources::block_t* head = gmp::resources::initialize_memory(pool_ptr, pool_size);
    ASSERT_NE(head, nullptr) << "Failed to initialize memory pool";
    
    gmp::resources::PinnedHostMemory pinned_host_mem(shared_mutex, head, 256);
    
    // Test allocation
    size_t test_size = 1024;
    void* ptr = pinned_host_mem.allocate(test_size);
    ASSERT_NE(ptr, nullptr) << "Pinned host memory allocation failed";
    std::cout << "Allocated " << test_size << " bytes in pinned host memory at " << ptr << std::endl;
    
    // Test deallocation
    pinned_host_mem.deallocate(ptr);
    std::cout << "Deallocated pinned host memory at " << ptr << std::endl;
    
    // Cleanup
    cudaFreeHost(pool_ptr);
}

TEST_F(RMMTest, ManagedMemoryAllocation) {
    std::cout << "Testing ManagedMemory allocation" << std::endl;
    
    // Allocate a large block of managed memory for testing
    size_t pool_size = 1024 * 1024; // 1MB
    void* pool_ptr = nullptr;
    cudaError_t cuda_status = cudaMallocManaged(&pool_ptr, pool_size);
    ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to allocate managed memory";
    
    std::mutex mutex;
    gmp::resources::block_t* head = gmp::resources::initialize_memory(pool_ptr, pool_size);
    ASSERT_NE(head, nullptr) << "Failed to initialize memory pool";
    
    gmp::resources::ManagedMemory managed_mem(head, 256, &mutex);
    
    // Test allocation
    size_t test_size = 1024;
    void* ptr = managed_mem.allocate(test_size);
    ASSERT_NE(ptr, nullptr) << "Managed memory allocation failed";
    std::cout << "Allocated " << test_size << " bytes in managed memory at " << ptr << std::endl;
    
    // Test deallocation
    managed_mem.deallocate(ptr);
    std::cout << "Deallocated managed memory at " << ptr << std::endl;
    
    // Cleanup
    cudaFree(pool_ptr);
}

TEST_F(RMMTest, PerformanceComparison) {
    std::cout << "Testing performance comparison" << std::endl;
    
    std::mutex mutex;
    gmp::resources::StreamedDeviceMemory streamed_dev_mem(&mutex, dmmr.get());
    streamed_dev_mem.insert_thread_stream_mapping(pthread_self(), stream);
    
    size_t test_size = 1024;
    const int num_iterations = 1000;
    
    // Time cudaMalloc/cudaFree
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        void* ptr = nullptr;
        cudaError_t status = cudaMalloc(&ptr, test_size);
        ASSERT_EQ(status, cudaSuccess) << "cudaMalloc failed";
        status = cudaFree(ptr);
        ASSERT_EQ(status, cudaSuccess) << "cudaFree failed";
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto cuda_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Time RMM allocation
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        void* ptr = streamed_dev_mem.allocate(test_size);
        ASSERT_NE(ptr, nullptr) << "RMM allocation failed";
        streamed_dev_mem.deallocate(ptr);
    }
    end = std::chrono::high_resolution_clock::now();
    auto rmm_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Performance comparison (" << num_iterations << " iterations):" << std::endl;
    std::cout << "cudaMalloc/cudaFree: " << cuda_duration << " microseconds" << std::endl;
    std::cout << "RMM allocate/deallocate: " << rmm_duration << " microseconds" << std::endl;
    std::cout << "RMM speedup: " << (double)cuda_duration / rmm_duration << "x" << std::endl;
    
    // RMM might not always be faster than direct CUDA allocation for small allocations
    // Just ensure it's not unreasonably slow (within 5x of direct allocation)
    EXPECT_LE(rmm_duration, cuda_duration * 5) << "RMM should not be unreasonably slow";
}

TEST_F(RMMTest, ThreadSafety) {
    std::cout << "Testing thread safety" << std::endl;
    
    std::mutex mutex;
    gmp::resources::StreamedDeviceMemory streamed_dev_mem(&mutex, dmmr.get());
    streamed_dev_mem.insert_thread_stream_mapping(pthread_self(), stream);
    
    const int num_threads = 4;
    const int allocations_per_thread = 100;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    auto worker = [&](int thread_id) {
        // Set up thread-to-stream mapping for this worker thread
        cudaStream_t worker_stream;
        cudaError_t cuda_status = cudaStreamCreate(&worker_stream);
        if (cuda_status != cudaSuccess) {
            return; // Skip this thread if stream creation fails
        }
        
        streamed_dev_mem.insert_thread_stream_mapping(pthread_self(), worker_stream);
        
        std::vector<void*> ptrs;
        ptrs.reserve(allocations_per_thread);
        
        for (int i = 0; i < allocations_per_thread; ++i) {
            size_t size = 1024 + (i % 1000); // Varying sizes
            void* ptr = streamed_dev_mem.allocate(size);
            if (ptr != nullptr) {
                ptrs.push_back(ptr);
            }
        }
        
        // Deallocate all
        for (void* ptr : ptrs) {
            streamed_dev_mem.deallocate(ptr);
        }
        
        // Clean up the stream
        cudaStreamDestroy(worker_stream);
        
        success_count++;
    };
    
    // Start threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(success_count.load(), num_threads) << "All threads should complete successfully";
    std::cout << "Thread safety test completed with " << success_count.load() << " successful threads" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 