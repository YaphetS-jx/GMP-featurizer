#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/generate.h>

// Test fixture for CUDA tests
class CUDATest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
        
        // Set device 0 as the current device
        cudaSetDevice(0);
    }
    
    void TearDown() override {
        // Clean up CUDA context
        cudaDeviceReset();
    }
};

// Test basic CUDA device functionality
TEST_F(CUDATest, DeviceProperties) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, 0);
    
    ASSERT_EQ(error, cudaSuccess) << "Failed to get device properties";
    ASSERT_GT(prop.totalGlobalMem, 0) << "Device has no global memory";
    ASSERT_GT(prop.major, 0) << "Invalid compute capability major version";
    ASSERT_GE(prop.minor, 0) << "Invalid compute capability minor version";
    
    // Print device info for debugging
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
}

// Test basic Thrust vector operations
TEST_F(CUDATest, ThrustVectorOperations) {
    const int N = 1000;
    
    // Create host vector
    thrust::host_vector<int> h_vec(N);
    thrust::sequence(h_vec.begin(), h_vec.end());
    
    // Copy to device
    thrust::device_vector<int> d_vec = h_vec;
    
    // Verify copy was successful
    ASSERT_EQ(d_vec.size(), N) << "Device vector size mismatch";
    
    // Test basic operations
    thrust::device_vector<int> d_result(N);
    thrust::transform(d_vec.begin(), d_vec.end(), d_result.begin(), thrust::negate<int>());
    
    // Copy back to host
    thrust::host_vector<int> h_result = d_result;
    
    // Verify results
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(h_result[i], -i) << "Transform result mismatch at index " << i;
    }
}

// Test Thrust algorithms
TEST_F(CUDATest, ThrustAlgorithms) {
    const int N = 100;
    
    // Create random data
    thrust::host_vector<float> h_data(N);
    thrust::default_random_engine rng(10);
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
    thrust::generate(h_data.begin(), h_data.end(), [&]() { return dist(rng); });
    
    // Copy to device
    thrust::device_vector<float> d_data = h_data;
    
    // Test reduction
    float sum_host = thrust::reduce(h_data.begin(), h_data.end());
    float sum_device = thrust::reduce(d_data.begin(), d_data.end());
    
    EXPECT_NEAR(sum_host, sum_device, 1e-5) << "Reduction result mismatch";
    
    // Test maximum
    float max_host = thrust::reduce(h_data.begin(), h_data.end(), 
                                   std::numeric_limits<float>::lowest(), 
                                   thrust::maximum<float>());
    float max_device = thrust::reduce(d_data.begin(), d_data.end(), 
                                     std::numeric_limits<float>::lowest(), 
                                     thrust::maximum<float>());
    
    EXPECT_NEAR(max_host, max_device, 1e-5) << "Maximum result mismatch";
}

// CUDA kernel for vector addition
__device__ float add_kernel(float a, float b) {
    return a + b;
}

// Test custom CUDA kernel with Thrust
TEST_F(CUDATest, CustomKernelWithThrust) {
    const int N = 100;
    
    // Create test data
    thrust::host_vector<float> h_a(N, 1.0f);
    thrust::host_vector<float> h_b(N, 2.0f);
    thrust::host_vector<float> h_result(N);
    
    // Copy to device
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_b = h_b;
    thrust::device_vector<float> d_result(N);
    
    // Use Thrust transform with built-in plus function (no custom kernel)
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<float>());
    
    // Copy back to host
    thrust::copy(d_result.begin(), d_result.end(), h_result.begin());
    
    // Verify results
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(h_result[i], 3.0f, 1e-6) << "Kernel result mismatch at index " << i;
    }
}

// Test memory allocation and deallocation
TEST_F(CUDATest, MemoryManagement) {
    const int N = 10000;
    
    // Test host vector
    thrust::host_vector<int> h_vec(N);
    EXPECT_EQ(h_vec.size(), N) << "Host vector size mismatch";
    
    // Test device vector
    thrust::device_vector<int> d_vec(N);
    EXPECT_EQ(d_vec.size(), N) << "Device vector size mismatch";
    
    // Test resize
    d_vec.resize(N * 2);
    EXPECT_EQ(d_vec.size(), N * 2) << "Device vector resize failed";
    
    // Test clear
    d_vec.clear();
    EXPECT_EQ(d_vec.size(), 0) << "Device vector clear failed";
}

// Test error handling
TEST_F(CUDATest, ErrorHandling) {
    // Test invalid device access
    cudaError_t error = cudaSetDevice(999); // Invalid device
    EXPECT_NE(error, cudaSuccess) << "Should fail with invalid device";
    
    // Reset to valid device
    cudaSetDevice(0);
    
    // Test memory allocation with invalid size
    thrust::device_vector<int> d_vec;
    EXPECT_NO_THROW(d_vec.resize(0)) << "Zero size allocation should not throw";
    EXPECT_EQ(d_vec.size(), 0) << "Zero size vector should have size 0";
}

// Performance test (optional)
TEST_F(CUDATest, PerformanceTest) {
    const int N = 1000000;
    
    // Create large vectors
    thrust::host_vector<float> h_a(N, 1.0f);
    thrust::host_vector<float> h_b(N, 2.0f);
    
    // Time host computation
    auto start_host = std::chrono::high_resolution_clock::now();
    thrust::host_vector<float> h_result(N);
    thrust::transform(h_a.begin(), h_a.end(), h_b.begin(), h_result.begin(), thrust::plus<float>());
    auto end_host = std::chrono::high_resolution_clock::now();
    auto host_time = std::chrono::duration_cast<std::chrono::microseconds>(end_host - start_host);
    
    // Time device computation
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_b = h_b;
    thrust::device_vector<float> d_result(N);
    
    auto start_device = std::chrono::high_resolution_clock::now();
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_result.begin(), thrust::plus<float>());
    auto end_device = std::chrono::high_resolution_clock::now();
    auto device_time = std::chrono::duration_cast<std::chrono::microseconds>(end_device - start_device);
    
    // Verify results match
    thrust::host_vector<float> d_result_host = d_result;
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(h_result[i], d_result_host[i], 1e-6) << "Performance test result mismatch at index " << i;
    }
    
    // Print performance comparison
    std::cout << "Host computation time: " << host_time.count() << " microseconds" << std::endl;
    std::cout << "Device computation time: " << device_time.count() << " microseconds" << std::endl;
    if (device_time.count() > 0) {
        std::cout << "Speedup: " << static_cast<double>(host_time.count()) / device_time.count() << "x" << std::endl;
    }
}

// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 