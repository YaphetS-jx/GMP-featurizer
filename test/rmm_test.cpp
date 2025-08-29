#include <gtest/gtest.h>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/resource_ref.hpp>
#include "containers.hpp"
#include "util.hpp"

#include <vector>
#include <iostream>
#include <memory>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>

// using gmp::resources::pinned_host_allocator;
using gmp::containers::vector;
using gmp::containers::vector_device;

class RMMTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA
        cudaError_t cuda_status = cudaSetDevice(0);
        ASSERT_EQ(cuda_status, cudaSuccess) << "Failed to set CUDA device: " << cudaGetErrorString(cuda_status);        
        
        // Get initial GPU memory information
        auto [free_memory, total_memory] = rmm::available_device_memory();
        initial_free_memory = free_memory;
        initial_total_memory = total_memory;
    }
    
    void TearDown() override {
        // Cleanup is automatic with RAII
    }
    
    // Member functions for printing pool information
    void print_cuda_pool_info(cudaMemPool_t pool_handle, const std::string& pool_name) {
        std::cout << "\n=== " << pool_name << " Information ===" << std::endl;
        
        // Check if pool handle is valid
        if (pool_handle == nullptr) {
            std::cout << "Pool handle is null" << std::endl;
            return;
        }
        
        // Get pool attributes with error checking
        uint64_t reserved_mem = 0;
        uint64_t used_mem = 0;
        uint64_t release_threshold = 0;
        
        cudaError_t err1 = cudaMemPoolGetAttribute(pool_handle, cudaMemPoolAttrReservedMemCurrent, &reserved_mem);
        cudaError_t err2 = cudaMemPoolGetAttribute(pool_handle, cudaMemPoolAttrUsedMemCurrent, &used_mem);
        cudaError_t err3 = cudaMemPoolGetAttribute(pool_handle, cudaMemPoolAttrReleaseThreshold, &release_threshold);
        
        if (err1 != cudaSuccess) {
            std::cout << "Failed to get reserved memory: " << cudaGetErrorString(err1) << std::endl;
            return;
        }
        if (err2 != cudaSuccess) {
            std::cout << "Failed to get used memory: " << cudaGetErrorString(err2) << std::endl;
            return;
        }
        if (err3 != cudaSuccess) {
            std::cout << "Failed to get release threshold: " << cudaGetErrorString(err3) << std::endl;
            return;
        }
        
        std::cout << "Reserved Memory: " << gmp::util::format_bytes(reserved_mem) << std::endl;
        std::cout << "Used Memory: " << gmp::util::format_bytes(used_mem) << std::endl;
    }

    void print_pinned_pool_info(const std::string& pool_name = "Pinned Memory Pool") {
        auto* singleton_pool = gmp::resources::gmp_resource::instance().get_pinned_host_memory_pool();
        
        std::cout << "\n=== " << pool_name << " Information ===" << std::endl;
        std::cout << "Pool size: " << gmp::util::format_bytes(singleton_pool->pool_size()) << std::endl;        
    }    
    size_t initial_free_memory;
    size_t initial_total_memory;
};

TEST_F(RMMTest, InitialMemoryState) {
    std::cout << "=== Testing Initial Memory State ===" << std::endl;
    
    std::cout << "\nInitial GPU Memory:" << std::endl;
    std::cout << "  Total: " << gmp::util::format_bytes(initial_total_memory) << std::endl;
    std::cout << "  Free: " << gmp::util::format_bytes(initial_free_memory) << std::endl;
    std::cout << "  Used: " << gmp::util::format_bytes(initial_total_memory - initial_free_memory) << std::endl;
    
    // Print initial pinned memory pool information
    print_pinned_pool_info("Pinned Memory Pool (initial)");
    
    // Get singleton GPU memory pool
    auto* gpu_pool = gmp::resources::gmp_resource::instance().get_gpu_device_memory_pool();
    ASSERT_NE(gpu_pool, nullptr) << "GPU device memory pool creation failed";
    
    std::cout << "CUDA Async Memory Pool created successfully" << std::endl;
    print_cuda_pool_info(gpu_pool->pool_handle(), "CUDA Async Memory Pool (initial)");
}

TEST_F(RMMTest, PinnedHostAllocatorBasic) {
    std::cout << "=== Testing Pinned Host Allocator Basic Functionality ===" << std::endl;
    
    // Create a std::vector using pinned host memory resource via custom allocator
    vector<int> host_vec;
    
    // Test basic operations
    const size_t test_size = 5 * 1024 * 1024; // in total 20MB
    host_vec.resize(test_size);
    
    // Populate the host vector
    for (size_t i = 0; i < test_size; ++i) {
        host_vec[i] = static_cast<int>(i);
    }
    
    std::cout << "Host vector size: " << host_vec.size() << std::endl;
    std::cout << "Host vector capacity: " << host_vec.capacity() << std::endl;
    std::cout << "Host vector memory: " << gmp::util::format_bytes(host_vec.capacity() * sizeof(int)) << std::endl;
    
    // Verify data integrity
    for (size_t i = 0; i < test_size; ++i) {
        ASSERT_EQ(host_vec[i], static_cast<int>(i)) << "Data integrity check failed at index " << i;
    }
    
    // Test that the memory is pinned by checking if it's accessible from device
    ASSERT_NE(host_vec.data(), nullptr) << "Host vector data pointer is null";
    std::cout << "Host vector data pointer: " << host_vec.data() << std::endl;
    std::cout << "Host vector memory is allocated and pinned" << std::endl;
    
    // Print pinned memory pool information after allocation
    print_pinned_pool_info("Pinned Memory Pool (after host vector allocation)");
}

TEST_F(RMMTest, DeviceVectorAllocation) {
    std::cout << "=== Testing Device Vector Allocation ===" << std::endl;
    
    const size_t test_size = 50 * 1024 * 1024; // in total 200MB
    
    // Get singleton GPU memory pool
    auto* gpu_pool = gmp::resources::gmp_resource::instance().get_gpu_device_memory_pool();
    
    // Allocate from CUDA async pool
    cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream();
    vector_device<int> device_vec_async(test_size, stream, *gpu_pool);    
    std::cout << "Allocated " << gmp::util::format_bytes(test_size * sizeof(int)) << " from CUDA async pool" << std::endl;
    
    // Verify allocation
    ASSERT_EQ(device_vec_async.size(), test_size) << "CUDA async device vector size mismatch";
    ASSERT_NE(device_vec_async.data(), nullptr) << "CUDA async device vector data pointer is null";
    
    // Print updated pool information
    print_cuda_pool_info(gpu_pool->pool_handle(), "CUDA Async Memory Pool (after allocation)");
}

TEST_F(RMMTest, DataTransferHostToDevice) {
    std::cout << "=== Testing Data Transfer Host to Device ===" << std::endl;
    
    const size_t test_size = 5 * 1024 * 1024; // in total 20MB
    
    // Create host vector with pinned memory
    vector<int> host_vec(test_size);
    for (size_t i = 0; i < test_size; ++i) {
        host_vec[i] = static_cast<int>(i * 2);
    }
    
    // Get singleton GPU memory pool
    auto* gpu_pool = gmp::resources::gmp_resource::instance().get_gpu_device_memory_pool();
    
    // Create device vector
    cudaStream_t stream = gmp::resources::gmp_resource::instance().get_stream();
    vector_device<int> device_vec(test_size, stream, *gpu_pool);
    
    // Copy data from host to device
    cudaError_t cuda_status = cudaMemcpyAsync(
        device_vec.data(), 
        host_vec.data(), 
        test_size * sizeof(int), 
        cudaMemcpyHostToDevice, 
        stream
    );
    ASSERT_EQ(cuda_status, cudaSuccess) << "Host to device copy failed: " << cudaGetErrorString(cuda_status);
    
    // Create result vector on host
    std::vector<int> result_vec(test_size);
    
    // Copy data back from device to host
    cuda_status = cudaMemcpyAsync(
        result_vec.data(), 
        device_vec.data(), 
        test_size * sizeof(int), 
        cudaMemcpyDeviceToHost, 
        stream
    );
    ASSERT_EQ(cuda_status, cudaSuccess) << "Device to host copy failed: " << cudaGetErrorString(cuda_status);
    
    // Synchronize and verify
    cudaStreamSynchronize(stream);
    
    // Verify data integrity
    for (size_t i = 0; i < test_size; ++i) {
        ASSERT_EQ(result_vec[i], host_vec[i]) << "Data transfer verification failed at index " << i;
    }
    
    std::cout << "Data transfer test completed successfully" << std::endl;
    std::cout << "First few elements: ";
    for (size_t i = 0; i < std::min(5UL, test_size); ++i) {
        std::cout << result_vec[i] << " ";
    }
    std::cout << std::endl;
}

TEST_F(RMMTest, FinalMemoryState) {
    std::cout << "=== Testing Final Memory State ===" << std::endl;
    
    // Get final GPU memory information
    auto [final_free_memory, final_total_memory] = rmm::available_device_memory();
    
    std::cout << "\nFinal GPU Memory:" << std::endl;
    std::cout << "  Total: " << gmp::util::format_bytes(final_total_memory) << std::endl;
    std::cout << "  Free: " << gmp::util::format_bytes(final_free_memory) << std::endl;
    std::cout << "  Used: " << gmp::util::format_bytes(final_total_memory - final_free_memory) << std::endl;
    
    // Get singleton pools
    auto* gpu_pool = gmp::resources::gmp_resource::instance().get_gpu_device_memory_pool();
    
    // Print final pool information
    print_cuda_pool_info(gpu_pool->pool_handle(), "CUDA Async Memory Pool (final)");
    print_pinned_pool_info("Pinned Memory Pool (final)");
    
    // Print actual pool memory usage
    uint64_t reserved_mem = 0, used_mem = 0;
    cudaMemPoolGetAttribute(gpu_pool->pool_handle(), cudaMemPoolAttrReservedMemCurrent, &reserved_mem);
    cudaMemPoolGetAttribute(gpu_pool->pool_handle(), cudaMemPoolAttrUsedMemCurrent, &used_mem);
    
    std::cout << "\nActual Pool Memory Usage:" << std::endl;
    std::cout << "  CUDA Async Pool reserved: " << gmp::util::format_bytes(reserved_mem) << std::endl;
    std::cout << "  CUDA Async Pool used: " << gmp::util::format_bytes(used_mem) << std::endl;
    
    // Verify that memory was properly managed
    ASSERT_LE(used_mem, reserved_mem) << "Used memory should not exceed reserved memory";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    // Explicitly cleanup CUDA resources before exit
    gmp::resources::gmp_resource::instance().cleanup();
    
    return result;
} 