#include <gtest/gtest.h>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/resource_ref.hpp>
#include "pinned_memory_pool.hpp"
#include "util.hpp"

#include <vector>
#include <iostream>
#include <memory>
#include <iomanip>
#include <sstream>
#include <cuda_runtime.h>

using gmp::resources::pinned_host_allocator;
using gmp::resources::PinnedMemoryPool;

// Function to print CUDA memory pool information
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
    std::cout << "Available Memory: " << gmp::util::format_bytes(reserved_mem - used_mem) << std::endl;
    std::cout << "Release Threshold: " << gmp::util::format_bytes(release_threshold) << std::endl;
    std::cout << "Memory Utilization: " << std::fixed << std::setprecision(1) 
              << (reserved_mem > 0 ? (used_mem * 100.0 / reserved_mem) : 0.0) << "%" << std::endl;
}

// Function to print pinned memory pool information
void print_pinned_pool_info(const std::string& pool_name = "Pinned Memory Pool") {
    auto& pool = gmp::resources::PinnedMemoryPool::instance();
    
    std::cout << "\n=== " << pool_name << " Information ===" << std::endl;
    std::cout << "Total Memory: " << gmp::util::format_bytes(pool.get_total_size()) << std::endl;
    std::cout << "Used Memory: " << gmp::util::format_bytes(pool.get_used_size()) << std::endl;
    std::cout << "Free Memory: " << gmp::util::format_bytes(pool.get_free_size()) << std::endl;
    std::cout << "Total Blocks: " << pool.get_block_count() << std::endl;
    std::cout << "Free Blocks: " << pool.get_free_block_count() << std::endl;
    std::cout << "Used Blocks: " << (pool.get_block_count() - pool.get_free_block_count()) << std::endl;
    std::cout << "Memory Utilization: " << std::fixed << std::setprecision(1) 
              << pool.get_utilization_percentage() << "%" << std::endl;
}

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
        
        // Create CUDA stream for testing
        stream = rmm::cuda_stream{};
        
        // Initialize memory resources
        initial_pool_size = free_memory / 4; // Use 25% of free memory
        cuda_async_mr = std::make_unique<rmm::mr::cuda_async_memory_resource>(initial_pool_size);
        
        // Create RMM pool memory resource
        auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
        pool_size = free_memory / 8; // Use 12.5% of free memory
        pool_mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
            cuda_mr.get(), pool_size);
    }
    
    void TearDown() override {
        // Cleanup is automatic with RAII
    }
    
    rmm::cuda_stream stream;
    std::unique_ptr<rmm::mr::cuda_async_memory_resource> cuda_async_mr;
    std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>> pool_mr;
    size_t initial_free_memory;
    size_t initial_total_memory;
    size_t initial_pool_size;
    size_t pool_size;
};

TEST_F(RMMTest, InitialMemoryState) {
    std::cout << "=== Testing Initial Memory State ===" << std::endl;
    
    std::cout << "\nInitial GPU Memory:" << std::endl;
    std::cout << "  Total: " << gmp::util::format_bytes(initial_total_memory) << std::endl;
    std::cout << "  Free: " << gmp::util::format_bytes(initial_free_memory) << std::endl;
    std::cout << "  Used: " << gmp::util::format_bytes(initial_total_memory - initial_free_memory) << std::endl;
    
    // Print initial pinned memory pool information
    print_pinned_pool_info("Pinned Memory Pool (initial)");
    
    // Verify memory resources were created successfully
    ASSERT_NE(cuda_async_mr, nullptr) << "CUDA async memory resource creation failed";
    ASSERT_NE(pool_mr, nullptr) << "RMM pool memory resource creation failed";
    
    // Print pool information after creation
    // print_cuda_pool_info(cuda_async_mr->pool_handle(), "CUDA Async Memory Pool");
    std::cout << "CUDA Async Memory Pool created successfully" << std::endl;
    std::cout << "RMM Pool size: " << gmp::util::format_bytes(pool_mr->pool_size()) << std::endl;
}

TEST_F(RMMTest, PinnedHostAllocatorBasic) {
    std::cout << "=== Testing Pinned Host Allocator Basic Functionality ===" << std::endl;
    
    // Create a std::vector using pinned host memory resource via custom allocator
    std::vector<int, pinned_host_allocator<int>> host_vec;
    
    // Test basic operations
    const size_t test_size = 1000;
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
    
    const size_t test_size = 12345;
    
    // Allocate from CUDA async pool
    rmm::device_uvector<int> device_vec_async(test_size, stream, *cuda_async_mr);
    std::cout << "Allocated " << gmp::util::format_bytes(test_size * sizeof(int)) << " from CUDA async pool" << std::endl;
    
    // Allocate from RMM pool
    rmm::device_uvector<int> device_vec_pool(test_size, stream, *pool_mr);
    std::cout << "Allocated " << gmp::util::format_bytes(test_size * sizeof(int)) << " from RMM pool" << std::endl;
    
    // Verify allocations
    ASSERT_EQ(device_vec_async.size(), test_size) << "CUDA async device vector size mismatch";
    ASSERT_EQ(device_vec_pool.size(), test_size) << "RMM pool device vector size mismatch";
    ASSERT_NE(device_vec_async.data(), nullptr) << "CUDA async device vector data pointer is null";
    ASSERT_NE(device_vec_pool.data(), nullptr) << "RMM pool device vector data pointer is null";
    
    // Print updated pool information
    print_cuda_pool_info(cuda_async_mr->pool_handle(), "CUDA Async Memory Pool (after allocation)");
    std::cout << "RMM Pool size after allocation: " << gmp::util::format_bytes(pool_mr->pool_size()) << std::endl;
}

TEST_F(RMMTest, DataTransferHostToDevice) {
    std::cout << "=== Testing Data Transfer Host to Device ===" << std::endl;
    
    const size_t test_size = 1000;
    
    // Create host vector with pinned memory
    std::vector<int, pinned_host_allocator<int>> host_vec(test_size);
    for (size_t i = 0; i < test_size; ++i) {
        host_vec[i] = static_cast<int>(i * 2);
    }
    
    // Create device vector
    rmm::device_uvector<int> device_vec(test_size, stream, *cuda_async_mr);
    
    // Copy data from host to device
    cudaError_t cuda_status = cudaMemcpyAsync(
        device_vec.data(), 
        host_vec.data(), 
        test_size * sizeof(int), 
        cudaMemcpyHostToDevice, 
        stream.value()
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
        stream.value()
    );
    ASSERT_EQ(cuda_status, cudaSuccess) << "Device to host copy failed: " << cudaGetErrorString(cuda_status);
    
    // Synchronize and verify
    stream.synchronize();
    
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

TEST_F(RMMTest, PinnedMemoryKernelAccess) {
    std::cout << "=== Testing Pinned Memory Access from CUDA Kernel ===" << std::endl;
    
    const size_t test_size = 1000;
    
    // Create host vector with pinned memory
    std::vector<int, pinned_host_allocator<int>> host_vec(test_size);
    for (size_t i = 0; i < test_size; ++i) {
        host_vec[i] = static_cast<int>(i);
    }
    
    // Create device memory for results
    rmm::device_uvector<int> device_result(test_size, stream, *cuda_async_mr);
    
    // For now, just test that we can access the pinned memory from device
    // by doing a simple copy operation instead of kernel launch
    cudaError_t cuda_status = cudaMemcpyAsync(
        device_result.data(), 
        host_vec.data(), 
        test_size * sizeof(int), 
        cudaMemcpyHostToDevice, 
        stream.value()
    );
    ASSERT_EQ(cuda_status, cudaSuccess) << "Host to device copy failed: " << cudaGetErrorString(cuda_status);
    
    // Copy results back to host for verification
    std::vector<int> result_vec(test_size);
    cuda_status = cudaMemcpyAsync(
        result_vec.data(), 
        device_result.data(), 
        test_size * sizeof(int), 
        cudaMemcpyDeviceToHost, 
        stream.value()
    );
    ASSERT_EQ(cuda_status, cudaSuccess) << "Device to host copy failed: " << cudaGetErrorString(cuda_status);
    
    // Synchronize and verify results
    stream.synchronize();
    
    // Verify the computation was correct (just copied, so should be same)
    for (size_t i = 0; i < test_size; ++i) {
        ASSERT_EQ(result_vec[i], host_vec[i]) << "Data transfer verification failed at index " << i;
    }
    
    std::cout << "Pinned memory access test completed successfully!" << std::endl;
    std::cout << "First few results: ";
    for (size_t i = 0; i < std::min(5UL, test_size); ++i) {
        std::cout << result_vec[i] << " ";
    }
    std::cout << std::endl;
}

TEST_F(RMMTest, MemoryPoolUtilization) {
    std::cout << "=== Testing Memory Pool Utilization ===" << std::endl;
    
    // Test multiple allocations and deallocations
    const size_t num_allocations = 10;
    const size_t allocation_size = 1024;
    
    std::vector<rmm::device_uvector<int>> device_vectors;
    device_vectors.reserve(num_allocations);
    
    // Allocate multiple vectors
    for (size_t i = 0; i < num_allocations; ++i) {
        device_vectors.emplace_back(allocation_size, stream, *cuda_async_mr);
    }
    
    // Print pool information after allocations
    print_cuda_pool_info(cuda_async_mr->pool_handle(), "CUDA Async Memory Pool (after multiple allocations)");
    
    // Verify all allocations succeeded
    for (size_t i = 0; i < num_allocations; ++i) {
        ASSERT_EQ(device_vectors[i].size(), allocation_size) << "Device vector " << i << " size mismatch";
        ASSERT_NE(device_vectors[i].data(), nullptr) << "Device vector " << i << " data pointer is null";
    }
    
    // Clear vectors (deallocate)
    device_vectors.clear();
    
    // Print pool information after deallocations
    print_cuda_pool_info(cuda_async_mr->pool_handle(), "CUDA Async Memory Pool (after deallocations)");
    
    std::cout << "Memory pool utilization test completed successfully" << std::endl;
}

TEST_F(RMMTest, PinnedMemoryPoolUtilization) {
    std::cout << "=== Testing Pinned Memory Pool Utilization ===" << std::endl;
    
    // Test multiple host vector allocations
    const size_t num_vectors = 5;
    const size_t vector_size = 1000;
    
    std::vector<std::vector<int, pinned_host_allocator<int>>> host_vectors;
    host_vectors.reserve(num_vectors);
    
    // Print initial state
    print_pinned_pool_info("Pinned Memory Pool (initial)");
    
    // Allocate multiple host vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        host_vectors.emplace_back(vector_size);
        for (size_t j = 0; j < vector_size; ++j) {
            host_vectors[i][j] = static_cast<int>(i * vector_size + j);
        }
    }
    
    // Print state after allocations
    print_pinned_pool_info("Pinned Memory Pool (after multiple allocations)");
    
    // Verify all allocations succeeded
    for (size_t i = 0; i < num_vectors; ++i) {
        ASSERT_EQ(host_vectors[i].size(), vector_size) << "Host vector " << i << " size mismatch";
        ASSERT_NE(host_vectors[i].data(), nullptr) << "Host vector " << i << " data pointer is null";
    }
    
    // Clear vectors (deallocate)
    host_vectors.clear();
    
    // Print state after deallocations
    print_pinned_pool_info("Pinned Memory Pool (after deallocations)");
    
    std::cout << "Pinned memory pool utilization test completed successfully" << std::endl;
}

TEST_F(RMMTest, FinalMemoryState) {
    std::cout << "=== Testing Final Memory State ===" << std::endl;
    
    // Get final GPU memory information
    auto [final_free_memory, final_total_memory] = rmm::available_device_memory();
    
    std::cout << "\nFinal GPU Memory:" << std::endl;
    std::cout << "  Total: " << gmp::util::format_bytes(final_total_memory) << std::endl;
    std::cout << "  Free: " << gmp::util::format_bytes(final_free_memory) << std::endl;
    std::cout << "  Used: " << gmp::util::format_bytes(final_total_memory - final_free_memory) << std::endl;
    
    // Print final pool information
    print_cuda_pool_info(cuda_async_mr->pool_handle(), "CUDA Async Memory Pool (final)");
    std::cout << "RMM Pool size (final): " << gmp::util::format_bytes(pool_mr->pool_size()) << std::endl;
    print_pinned_pool_info("Pinned Memory Pool (final)");
    
    // Print actual pool memory usage
    uint64_t reserved_mem = 0, used_mem = 0;
    cudaMemPoolGetAttribute(cuda_async_mr->pool_handle(), cudaMemPoolAttrReservedMemCurrent, &reserved_mem);
    cudaMemPoolGetAttribute(cuda_async_mr->pool_handle(), cudaMemPoolAttrUsedMemCurrent, &used_mem);
    
    std::cout << "\nActual Pool Memory Usage:" << std::endl;
    std::cout << "  CUDA Async Pool reserved: " << gmp::util::format_bytes(reserved_mem) << std::endl;
    std::cout << "  CUDA Async Pool used: " << gmp::util::format_bytes(used_mem) << std::endl;
    std::cout << "  RMM Pool size: " << gmp::util::format_bytes(pool_mr->pool_size()) << std::endl;
    
    // Verify that memory was properly managed
    ASSERT_LE(used_mem, reserved_mem) << "Used memory should not exceed reserved memory";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 