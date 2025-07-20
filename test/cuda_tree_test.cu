#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <random>
#include <set>
#include <tuple>
#include <algorithm>
#include "cuda_tree.hpp"
#include "containers.hpp"
#include "common_types.hpp"
#include "morton_codes.hpp"

using namespace gmp::tree;
using namespace gmp::tree::morton_codes;
using namespace gmp::containers;

// Helper function to compare two internal nodes
bool compare_nodes(const internal_node_t<uint32_t, int32_t>& a, const internal_node_t<uint32_t, int32_t>& b) {
    return a.left == b.left &&
           a.right == b.right &&
           a.lower_bound == b.lower_bound &&
           a.upper_bound == b.upper_bound;
}

// Helper function to print node details for debugging
std::string node_to_string(const internal_node_t<uint32_t, int32_t>& node) {
    std::stringstream ss;
    ss << "Node {" 
       << "left=" << node.left 
       << ", right=" << node.right 
       << ", lower_bound=" << node.lower_bound 
       << ", upper_bound=" << node.upper_bound 
       << "}";
    return ss.str();
}

class CudaTreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test data
        stream = gmp::resources::gmp_resource::instance().get_stream();
    }

    void TearDown() override {}
    cudaStream_t stream;
};

TEST_F(CudaTreeTest, ConstructorWithMortonCodes) {
    // Create some test morton codes
    vector_host<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    
    vector_device<uint32_t> morton_codes_device(morton_codes.size(), stream);
    cudaMemcpyAsync(morton_codes_device.data(), morton_codes.data(), morton_codes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    cuda_binary_radix_tree_t<uint32_t, int32_t> tree(morton_codes_device, 9, stream);
    
    // Synchronize before accessing results
    cudaStreamSynchronize(stream);
    
    // verify all the internal nodes 
    auto internal_nodes = tree.get_internal_nodes();
    EXPECT_EQ(internal_nodes.size(), 4);

    std::array<internal_node_t<uint32_t, int32_t>, 4> benchmark = {{
        {8, 4, 0, 511},
        {0, 1, 0, 7},
        {2, 3, 64, 65},
        {6, 7, 0, 127}
    }};

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(compare_nodes(internal_nodes[i], benchmark[i])) << "Node " << i << " mismatch: " << node_to_string(internal_nodes[i]);
    }
}

TEST_F(CudaTreeTest, TraverseBasicTest) {
    // Test morton codes
    vector_host<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 3;

    // Create device morton codes
    vector_device<uint32_t> morton_codes_device(morton_codes.size(), stream);
    cudaMemcpyAsync(morton_codes_device.data(), morton_codes.data(), morton_codes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // Create and build the tree
    cuda_binary_radix_tree_t<uint32_t, int32_t> tree(morton_codes_device, num_bits * 3, stream);
    
    // Synchronize before proceeding
    cudaStreamSynchronize(stream);

    // Define query bounds
    uint32_t query_lower_bound = interleave_bits(0, 0, 0, num_bits);
    uint32_t query_upper_bound = interleave_bits(0b100, 0b100, 0b100, num_bits);

    // Perform traversal
    auto result = tree.traverse(query_lower_bound, query_upper_bound, stream);
    
    // Synchronize to get results
    cudaStreamSynchronize(stream);
    
    // Copy results to host
    vector_host<int32_t> result_indices_host(result.indices.size());
    vector_host<int32_t> result_shifts_host(result.shifts.size());
    
    cudaMemcpyAsync(result_indices_host.data(), result.indices.data(), 
        result.indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(result_shifts_host.data(), result.shifts.data(), 
        result.shifts.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Verify results
    EXPECT_EQ(result_indices_host.size(), 3);
    EXPECT_EQ(result_shifts_host.size(), 9); // 3 results * 3 int32s per result
    
    // Check that we got the expected indices (0, 1, 2)
    std::vector<int32_t> expected_indices = {0, 1, 2};
    std::sort(result_indices_host.begin(), result_indices_host.end());
    std::sort(expected_indices.begin(), expected_indices.end());
    
    EXPECT_TRUE(std::equal(result_indices_host.begin(), result_indices_host.end(), expected_indices.begin()));
    
    // Check that all shifts are 0
    for (int i = 0; i < result_shifts_host.size(); ++i) {
        EXPECT_EQ(result_shifts_host[i], 0);
    }
}

TEST_F(CudaTreeTest, TraverseGeneralCase) {
    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15); // 2^4 - 1 = 15

    // Generate random points
    const int num_bits = 4;
    const int num_points = 1000;

    // Generate points and morton codes
    std::set<std::tuple<int, int, int>> unique_points;
    vector_host<uint32_t> morton_codes;
    morton_codes.reserve(num_points);

    for (int i = 0; i < num_points; ++i) {
        int x = dis(gen);
        int y = dis(gen);
        int z = dis(gen);
        unique_points.insert({x, y, z});
    }

    // Convert to morton codes
    for (const auto& point : unique_points) {
        int x, y, z;
        std::tie(x, y, z) = point;
        morton_codes.push_back(interleave_bits(x, y, z, num_bits));
    }
    
    std::cout << "num of unique points: " << unique_points.size() << std::endl;

    // Sort morton codes
    std::sort(morton_codes.begin(), morton_codes.end());

    // Create device morton codes
    vector_device<uint32_t> morton_codes_device(morton_codes.size(), stream);
    cudaMemcpyAsync(morton_codes_device.data(), morton_codes.data(), morton_codes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // Build tree
    cuda_binary_radix_tree_t<uint32_t, int32_t> tree(morton_codes_device, num_bits * 3, stream);
    
    // Synchronize before proceeding
    cudaStreamSynchronize(stream);

    // Generate random query window
    int x1 = dis(gen), x2 = dis(gen);
    int y1 = dis(gen), y2 = dis(gen);
    int z1 = dis(gen), z2 = dis(gen);
    
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    if (z1 > z2) std::swap(z1, z2);

    uint32_t query_min = interleave_bits(x1, y1, z1, num_bits);
    uint32_t query_max = interleave_bits(x2, y2, z2, num_bits);

    // Get results from CUDA tree traversal
    auto result = tree.traverse(query_min, query_max, stream);
    
    // Synchronize to get results
    cudaStreamSynchronize(stream);
    
    // Copy results to host
    vector_host<int32_t> result_indices_host(result.indices.size());
    cudaMemcpyAsync(result_indices_host.data(), result.indices.data(), 
        result.indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Brute force check
    std::vector<int32_t> bench_result;
    uint32_t x_mask, y_mask, z_mask;
    create_masks(x_mask, y_mask, z_mask);
    
    for (size_t i = 0; i < morton_codes.size(); ++i) {
        if (mc_is_less_than_or_equal(query_min, morton_codes[i], x_mask, y_mask, z_mask) && 
            mc_is_less_than_or_equal(morton_codes[i], query_max, x_mask, y_mask, z_mask)) {
            bench_result.push_back(i);
        }
    }

    // Sort both results
    std::sort(result_indices_host.begin(), result_indices_host.end());
    std::sort(bench_result.begin(), bench_result.end());

    std::cout << "CUDA query counts: " << result_indices_host.size() << std::endl;
    std::cout << "CPU query counts: " << bench_result.size() << std::endl;

    // Compare results
    EXPECT_EQ(result_indices_host.size(), bench_result.size()) << "Result sizes don't match!";
    EXPECT_TRUE(std::equal(result_indices_host.begin(), result_indices_host.end(), bench_result.begin())) 
        << "Results don't match!";
}

TEST_F(CudaTreeTest, TraverseEmptyResult) {
    // Test with query that should return no results
    vector_host<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 3;

    // Create device morton codes
    vector_device<uint32_t> morton_codes_device(morton_codes.size(), stream);
    cudaMemcpyAsync(morton_codes_device.data(), morton_codes.data(), morton_codes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // Create and build the tree
    cuda_binary_radix_tree_t<uint32_t, int32_t> tree(morton_codes_device, num_bits * 3, stream);
    
    // Synchronize before proceeding
    cudaStreamSynchronize(stream);

    // Define query bounds that should return no results
    // Use bounds that are definitely outside the range of our morton codes
    uint32_t query_lower_bound = 0xFFFFFFFF; // Maximum value
    uint32_t query_upper_bound = 0xFFFFFFFF; // Maximum value

    // Perform traversal
    auto result = tree.traverse(query_lower_bound, query_upper_bound, stream);
    
    // Synchronize to get results
    cudaStreamSynchronize(stream);
    
    // Verify results
    EXPECT_EQ(result.indices.size(), 0);
    EXPECT_EQ(result.shifts.size(), 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    // Explicitly cleanup CUDA resources before exit
    gmp::resources::gmp_resource::instance().cleanup();
    
    return result;
} 