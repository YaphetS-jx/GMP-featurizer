#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <random>
#include <set>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <memory>
#include "cuda_tree.hpp"
#include "containers.hpp"
#include "morton_codes.hpp"
#include "tree.hpp"
#include "region_query.hpp"
#include "geometry.hpp"

using namespace gmp::tree;
using namespace gmp::tree::morton_codes;
using namespace gmp::containers;
using namespace gmp::region_query;
using namespace gmp::geometry;

// Helper function to compare two internal nodes
bool compare_nodes(const internal_node_t<int32_t, float>& a, const internal_node_t<int32_t, float>& b) {
    const float tolerance = 1e-6f;
    return a.get_left() == b.get_left() &&
           a.get_right() == b.get_right() &&
           std::abs(a.lower_bound_coords[0] - b.lower_bound_coords[0]) < tolerance &&
           std::abs(a.lower_bound_coords[1] - b.lower_bound_coords[1]) < tolerance &&
           std::abs(a.lower_bound_coords[2] - b.lower_bound_coords[2]) < tolerance &&
           std::abs(a.upper_bound_coords[0] - b.upper_bound_coords[0]) < tolerance &&
           std::abs(a.upper_bound_coords[1] - b.upper_bound_coords[1]) < tolerance &&
           std::abs(a.upper_bound_coords[2] - b.upper_bound_coords[2]) < tolerance;
}

// Helper function to print node details for debugging
std::string node_to_string(const internal_node_t<int32_t, float>& node) {
    std::stringstream ss;
    ss << "Node {" 
       << "left=" << node.get_left() 
       << ", right=" << node.get_right() 
       << ", lower_coords=[" << node.lower_bound_coords[0] << "," << node.lower_bound_coords[1] << "," << node.lower_bound_coords[2] << "]"
       << ", upper_coords=[" << node.upper_bound_coords[0] << "," << node.upper_bound_coords[1] << "," << node.upper_bound_coords[2] << "]"
       << "}";
    return ss.str();
}

class CudaTreeTest : public ::testing::Test {
protected:
    void SetUp() override {
        stream = gmp::resources::gmp_resource::instance().get_stream();
        gmp::resources::gmp_resource::instance().get_device_memory_manager();
        gmp::resources::gmp_resource::instance().get_pinned_memory_manager();

        // Check CUDA errors from previous operations
        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            std::cerr << "CUDA error in SetUp: " << cudaGetErrorString(cuda_status) << std::endl;
            FAIL();
        }
    }

    void TearDown() override {}
    cudaStream_t stream;
};


// Warp-level traverse test using cuda_tree_traverse_warp kernel
TEST_F(CudaTreeTest, WarpTraverseTest) {
    // Use simple test data that we know works
    vector<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 4; // 4 bits per dimension = 12 total bits
    const int num_queries = 32; // Single warp for simplicity

    // Build CUDA tree
    cuda_binary_radix_tree_t<int32_t, gmp::gmp_float> cuda_tree(morton_codes, num_bits * 3);
    cudaStreamSynchronize(stream);

    // Generate query positions (simple test points)
    vector<point3d_t<gmp::gmp_float>> query_positions(num_queries);
    vector<int32_t> query_target_indexes(num_queries);
    vector<array3d_t<int32_t>> query_cell_shifts(num_queries);
    
    for (int i = 0; i < num_queries; ++i) {
        query_positions[i] = {
            gmp::gmp_float(0.1f + i * 0.01f),
            gmp::gmp_float(0.2f + i * 0.01f),
            gmp::gmp_float(0.3f + i * 0.01f)
        };
        query_target_indexes[i] = i; // Each query targets itself
        query_cell_shifts[i] = {0, 0, 0}; // No cell shifts for this test
    }

    // Copy query data to device
    vector_device<point3d_t<gmp::gmp_float>> d_positions(num_queries, stream);
    vector_device<int32_t> d_query_target_indexes(num_queries, stream);
    vector_device<array3d_t<int32_t>> d_cell_shifts(num_queries, stream);
    
    cudaMemcpyAsync(d_positions.data(), query_positions.data(), 
                    num_queries * sizeof(point3d_t<gmp::gmp_float>), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_query_target_indexes.data(), query_target_indexes.data(), 
                    num_queries * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cell_shifts.data(), query_cell_shifts.data(), 
                    num_queries * sizeof(array3d_t<int32_t>), 
                    cudaMemcpyHostToDevice, stream);

    // Set up sphere check parameters
    cuda_check_sphere_t<uint32_t, gmp::gmp_float, int32_t> host_check_method;
    host_check_method.radius2 = 0.1f; // Match CPU radius
    host_check_method.num_bits_per_dim = num_bits;
    host_check_method.size_per_dim = 1.0f / (1 << (num_bits - 1));
    host_check_method.metric = gmp::math::sym_matrix3d_t<gmp::gmp_float>{
        {1.0f, 1.0f, 1.0f},  // diagonal elements
        {0.0f, 0.0f, 0.0f}   // off-diagonal elements
    };

    // Copy check method to constant memory
    cudaMemcpyToSymbol(check_sphere_constant, &host_check_method, sizeof(host_check_method));

    // Create traverse result
    cuda_traverse_result_t<int32_t> traverse_result(num_queries, stream);
    traverse_result.indexes.resize(morton_codes.size() * num_queries, stream); // Over-allocate

    // Set up kernel launch parameters
    constexpr int MAX_STACK = 24;
    const int threads_per_block = 256; // 8 warps per block
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = (num_queries + warps_per_block - 1) / warps_per_block;
    const size_t shmem_bytes = 2 * warps_per_block * MAX_STACK * sizeof(int32_t);

    dim3 block_size(threads_per_block);
    dim3 grid_size(num_blocks);

    std::cout << "Launching warp kernel with " << num_blocks << " blocks, " 
              << threads_per_block << " threads per block" << std::endl;

    // Launch the warp kernel
    cuda_tree_traverse_warp<uint32_t, gmp::gmp_float, int32_t, MAX_STACK><<<grid_size, block_size, shmem_bytes, stream>>>(
        cuda_tree.internal_nodes.data(), 
        cuda_tree.leaf_nodes.data(), 
        cuda_tree.num_leaf_nodes,
        d_positions.data(),
        d_query_target_indexes.data(),
        d_cell_shifts.data(),
        num_queries,
        traverse_result.indexes.data(),
        traverse_result.num_indexes.data(),
        traverse_result.num_indexes_offset.data()
    );

    // Check for kernel launch errors
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        FAIL() << "Kernel launch failed: " << cudaGetErrorString(kernel_error);
    }

    // Synchronize and copy results back
    cudaStreamSynchronize(stream);

    vector<int32_t> h_num_indexes(num_queries);
    cudaMemcpyAsync(h_num_indexes.data(), traverse_result.num_indexes.data(), 
                    num_queries * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Verify results
    int total_results = 0;
    for (int i = 0; i < num_queries; ++i) {
        total_results += h_num_indexes[i];
        EXPECT_GE(h_num_indexes[i], 0) << "Query " << i << " has negative result count";
    }

    std::cout << "Total results found: " << total_results << std::endl;
    std::cout << "Average results per query: " << (float)total_results / num_queries << std::endl;

    // Verify that we got some results (this depends on the random data)
    EXPECT_GT(total_results, 0) << "No results found for any query";

    // Test with a single warp (32 queries)
    const int single_warp_queries = 32;
    vector<point3d_t<gmp::gmp_float>> single_warp_positions(single_warp_queries);
    vector<int32_t> single_warp_target_indexes(single_warp_queries);
    vector<array3d_t<int32_t>> single_warp_cell_shifts(single_warp_queries);
    
    for (int i = 0; i < single_warp_queries; ++i) {
        single_warp_positions[i] = {
            gmp::gmp_float(0.5f + i * 0.01f),
            gmp::gmp_float(0.6f + i * 0.01f),
            gmp::gmp_float(0.7f + i * 0.01f)
        };
        single_warp_target_indexes[i] = i;
        single_warp_cell_shifts[i] = {0, 0, 0};
    }

    // Copy single warp data to device
    vector_device<point3d_t<gmp::gmp_float>> d_single_warp_positions(single_warp_queries, stream);
    vector_device<int32_t> d_single_warp_target_indexes(single_warp_queries, stream);
    vector_device<array3d_t<int32_t>> d_single_warp_cell_shifts(single_warp_queries, stream);
    
    cudaMemcpyAsync(d_single_warp_positions.data(), single_warp_positions.data(), 
                    single_warp_queries * sizeof(point3d_t<gmp::gmp_float>), 
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_single_warp_target_indexes.data(), single_warp_target_indexes.data(), 
                    single_warp_queries * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_single_warp_cell_shifts.data(), single_warp_cell_shifts.data(), 
                    single_warp_queries * sizeof(array3d_t<int32_t>), 
                    cudaMemcpyHostToDevice, stream);

    // Create single warp result
    cuda_traverse_result_t<int32_t> single_warp_result(single_warp_queries, stream);
    single_warp_result.indexes.resize(morton_codes.size() * single_warp_queries, stream);

    // Launch single warp kernel (1 block, 32 threads)
    cuda_tree_traverse_warp<uint32_t, gmp::gmp_float, int32_t, MAX_STACK><<<1, 32, shmem_bytes, stream>>>(
        cuda_tree.internal_nodes.data(), 
        cuda_tree.leaf_nodes.data(), 
        cuda_tree.num_leaf_nodes,
        d_single_warp_positions.data(),
        d_single_warp_target_indexes.data(),
        d_single_warp_cell_shifts.data(),
        single_warp_queries,
        single_warp_result.indexes.data(),
        single_warp_result.num_indexes.data(),
        single_warp_result.num_indexes_offset.data()
    );

    cudaStreamSynchronize(stream);

    vector<int32_t> h_single_warp_num_indexes(single_warp_queries);
    cudaMemcpyAsync(h_single_warp_num_indexes.data(), single_warp_result.num_indexes.data(), 
                    single_warp_queries * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
                                                                                                               
    int single_warp_total = 0;
    for (int i = 0; i < single_warp_queries; ++i) {
        single_warp_total += h_single_warp_num_indexes[i];
        EXPECT_GE(h_single_warp_num_indexes[i], 0) << "Single warp query " << i << " has negative result count";
    }

    std::cout << "Single warp total results: " << single_warp_total << std::endl;
    EXPECT_GT(single_warp_total, 0) << "No results found for single warp test";

    // CPU benchmark for comparison
    std::cout << "\n=== CPU Benchmark ===" << std::endl;
    
    // Create CPU tree with same morton codes
    gmp::tree::binary_radix_tree_t<int32_t, gmp::gmp_float> cpu_tree;
    cpu_tree.build_tree(morton_codes, num_bits * 3);
    
    // Create a simple lattice for the CPU sphere check
    matrix3d_t<gmp::gmp_float> lattice_vectors;
    lattice_vectors[0] = {1.0f, 0.0f, 0.0f};
    lattice_vectors[1] = {0.0f, 1.0f, 0.0f};
    lattice_vectors[2] = {0.0f, 0.0f, 1.0f};
    auto lattice = std::make_unique<lattice_t<gmp::gmp_float>>(lattice_vectors);
    array3d_bool periodicity = {true, true, true};
    
    // Test CPU traversal for each query position
    int cpu_total_results = 0;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_queries; ++i) {
        // Create CPU sphere check using the proper constructor
        check_sphere_t<gmp::gmp_float, int32_t> cpu_check(num_bits, periodicity, lattice.get());
        cpu_check.update_point_radius(query_positions[i], std::sqrt(0.1f)); // Use smaller radius to match GPU
        
        // Traverse CPU tree
        auto cpu_result = cpu_tree.traverse(cpu_check);
        
        // Count results
        int query_results = 0;
        for (const auto& [index, shifts] : cpu_result) {
            query_results += shifts.size();
        }
        cpu_total_results += query_results;
    }
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    
    std::cout << "CPU total results: " << cpu_total_results << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " microseconds" << std::endl;
    std::cout << "CPU average per query: " << (float)cpu_total_results / num_queries << std::endl;
    
    // Compare results (should be similar)
    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "GPU total results: " << total_results << std::endl;
    std::cout << "CPU total results: " << cpu_total_results << std::endl;
    EXPECT_EQ(total_results, cpu_total_results);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    // Explicitly cleanup CUDA resources before exit
    gmp::resources::gmp_resource::instance().cleanup();
    
    return result;
} 