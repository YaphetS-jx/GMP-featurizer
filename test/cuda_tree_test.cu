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
#include "tree.hpp"

using namespace gmp::tree;
using namespace gmp::tree::morton_codes;
using namespace gmp::containers;
using gmp::math::array3d_t;
using gmp::gmp_float;

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
       << ", min_bounds=[" << node.min_bounds[0] << ", " << node.min_bounds[1] << ", " << node.min_bounds[2] << "]"
       << ", max_bounds=[" << node.max_bounds[0] << ", " << node.max_bounds[1] << ", " << node.max_bounds[2] << "]"
       << "}";
    return ss.str();
}

template <typename FloatType>
array3d_t<FloatType> compute_min_bounds(uint32_t bound, int num_bits_per_dim) {
    uint32_t x, y, z;
    deinterleave_bits(bound, num_bits_per_dim, x, y, z);
    return {
        morton_code_to_coordinate<FloatType, int32_t, uint32_t>(x, num_bits_per_dim),
        morton_code_to_coordinate<FloatType, int32_t, uint32_t>(y, num_bits_per_dim),
        morton_code_to_coordinate<FloatType, int32_t, uint32_t>(z, num_bits_per_dim)
    };
}

template <typename FloatType>
array3d_t<FloatType> compute_max_bounds(uint32_t bound, int num_bits_per_dim) {
    auto mins = compute_min_bounds<FloatType>(bound, num_bits_per_dim);
    FloatType size_per_dim = FloatType(1) / FloatType(1 << (num_bits_per_dim - 1));
    return {mins[0] + size_per_dim, mins[1] + size_per_dim, mins[2] + size_per_dim};
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

template <typename MortonCodeType>
class check_intersect_box_t : public compare_op_t<MortonCodeType> {
public:
    MortonCodeType query_lower_bound, query_upper_bound;
    MortonCodeType x_mask, y_mask, z_mask;

    explicit check_intersect_box_t(MortonCodeType query_lower_bound, MortonCodeType query_upper_bound)
        : query_lower_bound(query_lower_bound), query_upper_bound(query_upper_bound) 
    {
        create_masks(x_mask, y_mask, z_mask);
    }

    bool operator()(MortonCodeType lower_bound, MortonCodeType upper_bound) const override
    {
        return mc_is_less_than_or_equal(query_lower_bound, upper_bound, x_mask, y_mask, z_mask) && 
                mc_is_less_than_or_equal(lower_bound, query_upper_bound, x_mask, y_mask, z_mask);
    }

    std::vector<array3d_int32> operator()(MortonCodeType morton_code) const override 
    {
        std::vector<array3d_int32> result;
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
    auto num_bits = 9;
    
    // Create and build the tree
    cuda_binary_radix_tree_t<uint32_t, int32_t> tree(morton_codes, num_bits);

    EXPECT_EQ(tree.num_leaf_nodes, morton_codes.size());

    using inode_t = internal_node_t<uint32_t, int32_t>;
    
    // Calculate the number of internal nodes (num_leaf_nodes - 1)
    size_t num_internal_nodes = morton_codes.size() - 1;
    vector<inode_t> h_internal_nodes(num_internal_nodes);
    tree.get_internal_nodes(h_internal_nodes);

    // Synchronize before reading results
    cudaStreamSynchronize(stream);

    // Define expected benchmark values
    std::array<internal_node_t<uint32_t, int32_t>, 4> benchmark = {{
        {8, 4, 0, 511},
        {0, 1, 0, 7},
        {2, 3, 64, 65},
        {6, 7, 0, 127}
    }};
    // Compare actual nodes with benchmark
    int num_bits_per_dim = num_bits / 3;
    for (int i = 0; i < num_internal_nodes; i++) {
        EXPECT_TRUE(compare_nodes(h_internal_nodes[i], benchmark[i]))
            << "Node " << i << " mismatch: " << node_to_string(h_internal_nodes[i]);
        auto expected_min = compute_min_bounds<gmp_float>(benchmark[i].lower_bound, num_bits_per_dim);
        auto expected_max = compute_max_bounds<gmp_float>(benchmark[i].upper_bound, num_bits_per_dim);
        for (int axis = 0; axis < 3; ++axis) {
            EXPECT_NEAR(static_cast<double>(h_internal_nodes[i].min_bounds[axis]), static_cast<double>(expected_min[axis]), 1e-6)
                << "Node " << i << " min axis " << axis;
            EXPECT_NEAR(static_cast<double>(h_internal_nodes[i].max_bounds[axis]), static_cast<double>(expected_max[axis]), 1e-6)
                << "Node " << i << " max axis " << axis;
        }
    }
}

__global__
void traverse_check_box_kernel(const cudaTextureObject_t internal_nodes_tex, const cudaTextureObject_t internal_bounds_tex,
    const cudaTextureObject_t internal_min_bounds_tex, const cudaTextureObject_t internal_max_bounds_tex,
    const cudaTextureObject_t leaf_nodes_tex, const cudaTextureObject_t leaf_min_bounds_tex, const cudaTextureObject_t leaf_max_bounds_tex,
    int32_t num_leaf_nodes, const check_box_t check_box,
    int32_t* indexes, int32_t* num_indexes)
{
    *num_indexes = 0;
    cuda_tree_traverse<check_box_t, uint32_t, float, int32_t>(
        internal_nodes_tex, internal_bounds_tex, internal_min_bounds_tex, internal_max_bounds_tex,
        leaf_nodes_tex, leaf_min_bounds_tex, leaf_max_bounds_tex, num_leaf_nodes, check_box,
        point3d_t<float>{0.0f, 0.0f, 0.0f}, array3d_t<int32_t>{0, 0, 0}, indexes, *num_indexes, 0);
    return;
}

TEST_F(CudaTreeTest, TraverseBasicTest) {
    // Test morton codes
    vector<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 3;

    // Create and build the tree
    cuda_binary_radix_tree_t<uint32_t, int32_t> tree(morton_codes, num_bits * 3);
    
    // Synchronize before proceeding
    cudaStreamSynchronize(stream);

    // Define query bounds
    uint32_t query_lower_bound = interleave_bits(0, 0, 0, num_bits);
    uint32_t query_upper_bound = interleave_bits(0b100, 0b100, 0b100, num_bits);

    // Perform traversal
    result_t result(1, stream);

    check_box_t check_box;
    create_masks(check_box.x_mask, check_box.y_mask, check_box.z_mask);
    check_box.query_lower_bound = query_lower_bound;
    check_box.query_upper_bound = query_upper_bound;

    traverse_check_box_kernel<<<1, 1, 0, stream>>>(tree.internal_nodes_tex, tree.internal_bounds_tex,
        tree.internal_min_bounds_tex, tree.internal_max_bounds_tex,
        tree.leaf_nodes_tex, tree.leaf_min_bounds_tex, tree.leaf_max_bounds_tex,
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
    const int num_bits = 4;
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
    binary_radix_tree_t<uint32_t, int32_t> cpu_tree;
    cpu_tree.build_tree(cpu_morton_codes, num_bits * 3);
    
    check_intersect_box_t<uint32_t> op(query_min, query_max);

    // Get results from tree traversal
    auto cpu_result = cpu_tree.traverse(op);

    /////////////////
    // CUDA result //
    /////////////////
    auto dm = gmp::resources::gmp_resource::instance().get_device_memory_manager();
    auto stream = gmp::resources::gmp_resource::instance().get_stream();
    
    cuda_binary_radix_tree_t<uint32_t, int32_t> cuda_tree(morton_codes, num_bits * 3);

    result_t cuda_result(1, stream);
    cuda_result.indexes.resize(unique_points.size(), stream);

    check_box_t check_box;
    create_masks(check_box.x_mask, check_box.y_mask, check_box.z_mask);
    check_box.query_lower_bound = query_min;
    check_box.query_upper_bound = query_max;
    
    traverse_check_box_kernel<<<1, 1, 0, stream>>>(cuda_tree.internal_nodes_tex, cuda_tree.internal_bounds_tex,
        cuda_tree.internal_min_bounds_tex, cuda_tree.internal_max_bounds_tex,
        cuda_tree.leaf_nodes_tex, cuda_tree.leaf_min_bounds_tex, cuda_tree.leaf_max_bounds_tex,
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