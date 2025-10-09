#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <random>
#include <set>
#include <tuple>
#include <algorithm>
#include <cmath>
#include "cuda_tree.hpp"
#include "containers.hpp"
#include "morton_codes.hpp"
#include "tree.hpp"

using namespace gmp::tree;
using namespace gmp::tree::morton_codes;
using namespace gmp::containers;

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

template <typename FloatType>
class check_intersect_box_t : public compare_op_t<FloatType> {
public:
    int num_bits_per_dim;
    uint32_t query_lower_bound, query_upper_bound;
    uint32_t x_mask, y_mask, z_mask;

    explicit check_intersect_box_t(int num_bits_per_dim, uint32_t query_lower_bound, uint32_t query_upper_bound)
        : num_bits_per_dim(num_bits_per_dim), 
          query_lower_bound(query_lower_bound), query_upper_bound(query_upper_bound) 
    {
        create_masks(x_mask, y_mask, z_mask);
    }

    bool operator()(const array3d_t<FloatType>& lower_coords, const array3d_t<FloatType>& upper_coords) const override
    {
        return true;
    }

    std::vector<array3d_int32> operator()(const array3d_t<FloatType>& lower_coords, FloatType size_per_dim) const override 
    {
        std::vector<array3d_int32> result;
        // Convert float coordinates to morton codes for comparison
        uint32_t x = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(lower_coords[0], num_bits_per_dim);
        uint32_t y = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(lower_coords[1], num_bits_per_dim);
        uint32_t z = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(lower_coords[2], num_bits_per_dim);
        
        // Reconstruct morton code from coordinate array
        uint32_t morton_code = interleave_bits(x, y, z, num_bits_per_dim);
        
        if (mc_is_less_than_or_equal(query_lower_bound, morton_code, x_mask, y_mask, z_mask) && 
            mc_is_less_than_or_equal(morton_code, query_upper_bound, x_mask, y_mask, z_mask)) 
        {
            result.push_back(array3d_int32{0, 0, 0});
        }
        return result;
    }
};

using check_box_t = cuda_check_intersect_box_t<uint32_t, float, int32_t>;
using result_t = cuda_traverse_result_t<int32_t>;

TEST_F(CudaTreeTest, ConstructorTest) {
    vector<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 12; // 12 bits total = 4 bits per dimension
    
    // Create and build the tree
    cuda_binary_radix_tree_t<int32_t, float> tree(morton_codes, num_bits);

    EXPECT_EQ(tree.num_leaf_nodes, morton_codes.size());

    using inode_t = internal_node_t<int32_t, float>;
    
    // Calculate the number of internal nodes (num_leaf_nodes - 1)
    size_t num_internal_nodes = morton_codes.size() - 1;
    vector<inode_t> h_internal_nodes(num_internal_nodes);
    tree.get_internal_nodes(h_internal_nodes);

    // Synchronize before reading results
    cudaStreamSynchronize(stream);

    // Define expected benchmark values - using new structure
    std::array<internal_node_t<int32_t, float>, 4> benchmark;
    
    // Create benchmark nodes with proper structure - updated to match actual output
    benchmark[0].lower_bound_coords = {0.0f, 0.0f, 0.0f};
    benchmark[0].upper_bound_coords = {1.0f, 1.0f, 1.0f};
    benchmark[0].set_indices(8, 4);
    
    benchmark[1].lower_bound_coords = {0.0f, 0.0f, 0.0f};
    benchmark[1].upper_bound_coords = {0.25f, 0.25f, 0.25f};
    benchmark[1].set_indices(0, 1);
    
    benchmark[2].lower_bound_coords = {0.5f, 0.0f, 0.0f};
    benchmark[2].upper_bound_coords = {0.75f, 0.125f, 0.125f};
    benchmark[2].set_indices(2, 3);
    
    benchmark[3].lower_bound_coords = {0.0f, 0.0f, 0.0f};
    benchmark[3].upper_bound_coords = {1.0f, 0.5f, 0.5f};
    benchmark[3].set_indices(6, 7);
    
    // Compare actual nodes with benchmark
    for (int i = 0; i < num_internal_nodes; i++) {
        EXPECT_TRUE(compare_nodes(h_internal_nodes[i], benchmark[i])) 
            << "Node " << i << " mismatch: " << node_to_string(h_internal_nodes[i]);
    }
}

__global__
void traverse_check_box_kernel(const internal_node_t<int32_t, float>* internal_nodes, const array3d_t<float>* leaf_nodes, 
    int32_t num_leaf_nodes, const check_box_t check_box,
    int32_t* indexes, int32_t* num_indexes)
{
    cuda_tree_traverse<check_box_t, uint32_t, float, int32_t>(
        internal_nodes, leaf_nodes, num_leaf_nodes, check_box,
        point3d_t<float>{0.0f, 0.0f, 0.0f}, array3d_t<int32_t>{0, 0, 0}, indexes, num_indexes, 0);
    return;
}

TEST_F(CudaTreeTest, TraverseBasicTest) {
    // Test morton codes
    vector<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 4;

    // Create and build the tree
    cuda_binary_radix_tree_t<int32_t, float> tree(morton_codes, num_bits * 3);
    
    // Synchronize before proceeding
    cudaStreamSynchronize(stream);

    // Define query bounds
    uint32_t query_lower_bound = interleave_bits(0, 0, 0, num_bits);
    uint32_t query_upper_bound = interleave_bits(0b100, 0b100, 0b100, num_bits);

    // Perform traversal
    result_t result(1, stream);

    check_box_t check_box;
    check_box.num_bits_per_dim = num_bits;
    create_masks(check_box.x_mask, check_box.y_mask, check_box.z_mask);
    check_box.query_lower_bound = query_lower_bound;
    check_box.query_upper_bound = query_upper_bound;

    traverse_check_box_kernel<<<1, 1, 0, stream>>>(tree.internal_nodes.data(), tree.leaf_nodes.data(), 
        tree.num_leaf_nodes, check_box, 
        result.indexes.data(), result.num_indexes.data());

    // Copy results to host
    vector<int32_t> h_num_indexes(1);
    cudaMemcpyAsync(h_num_indexes.data(), result.num_indexes.data(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Copy results to host for verification
    vector<int32_t> h_indexes(h_num_indexes[0]);

    cudaMemcpyAsync(h_indexes.data(), result.indexes.data(), h_num_indexes[0] * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Basic verification
    EXPECT_EQ(h_num_indexes[0], 3);
}

TEST_F(CudaTreeTest, Traverse_general_case) {
    // Set up random number generation
    std::random_device rd;
    auto seed = rd();
    // auto seed = 1020269916;
    std::cout << "seed: " << seed << std::endl;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, 15); // 2^4 - 1 = 15

    // Generate random points
    const int num_bits = 5;
    // const int max_coord = (1 << num_bits) - 1;
    const int num_points = 5000;

    // Generate points and morton codes
    std::set<std::tuple<int, int, int>> unique_points;
    vector<uint32_t> morton_codes;
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

    // Generate random query window
    int x1 = dis(gen), x2 = dis(gen);
    int y1 = dis(gen), y2 = dis(gen);
    int z1 = dis(gen), z2 = dis(gen);
    
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    if (z1 > z2) std::swap(z1, z2);

    uint32_t query_min = interleave_bits(x1, y1, z1, num_bits);
    uint32_t query_max = interleave_bits(x2, y2, z2, num_bits);

    // Convert morton_codes to the type expected by CPU tree  
    vector<uint32_t> cpu_morton_codes(morton_codes.begin(), morton_codes.end());
    
    ////////////////
    // CPU result //
    ////////////////
    binary_radix_tree_t<int32_t, gmp::gmp_float> cpu_tree;
    cpu_tree.build_tree(cpu_morton_codes, num_bits * 3);
    
    check_intersect_box_t<gmp::gmp_float> op(num_bits, query_min, query_max);

    // Get results from tree traversal
    auto cpu_result = cpu_tree.traverse(op);

    /////////////////
    // CUDA result //
    /////////////////
    auto dm = gmp::resources::gmp_resource::instance().get_device_memory_manager();
    auto stream = gmp::resources::gmp_resource::instance().get_stream();
    
    cuda_binary_radix_tree_t<int32_t, float> cuda_tree(morton_codes, num_bits * 3);

    result_t cuda_result(1, stream);
    cuda_result.indexes.resize(unique_points.size(), stream);

    check_box_t check_box;
    check_box.num_bits_per_dim = num_bits;
    create_masks(check_box.x_mask, check_box.y_mask, check_box.z_mask);
    check_box.query_lower_bound = query_min;
    check_box.query_upper_bound = query_max;
    
    traverse_check_box_kernel<<<1, 1, 0, stream>>>(cuda_tree.internal_nodes.data(), cuda_tree.leaf_nodes.data(), 
        cuda_tree.num_leaf_nodes, check_box, 
        cuda_result.indexes.data(), cuda_result.num_indexes.data());
    
    // Copy results to host
    vector<int32_t> h_num_indexes(1);
    cudaMemcpyAsync(h_num_indexes.data(), cuda_result.num_indexes.data(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    vector<int32_t> h_indexes(h_num_indexes[0]);
    cudaMemcpyAsync(h_indexes.data(), cuda_result.indexes.data(), h_num_indexes[0] * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
                                                                                                               
    // check reuslt 
    EXPECT_EQ(cpu_result.size(), h_num_indexes[0]);
    for (int i = 0; i < h_num_indexes[0]; i++) {
        EXPECT_TRUE(cpu_result.find(h_indexes[i]) != cpu_result.end());

    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    
    // Explicitly cleanup CUDA resources before exit
    gmp::resources::gmp_resource::instance().cleanup();
    
    return result;
} 