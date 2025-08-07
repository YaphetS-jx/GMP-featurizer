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

template <typename MortonCodeType, typename VecType = vector<array3d_int32>>
class check_intersect_box_t : public compare_op_t<MortonCodeType, VecType> {
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

    VecType operator()(MortonCodeType morton_code) const override 
    {
        VecType result;
        if (mc_is_less_than_or_equal(query_lower_bound, morton_code, x_mask, y_mask, z_mask) && 
            mc_is_less_than_or_equal(morton_code, query_upper_bound, x_mask, y_mask, z_mask)) 
        {
            result.push_back(array3d_int32{0, 0, 0});
        }
        return result;
    }
};

template <typename MortonCodeType, typename IndexType>
class cuda_check_intersect_box_t : public cuda_compare_op_t<MortonCodeType, IndexType> {
public:
    MortonCodeType query_lower_bound, query_upper_bound;
    MortonCodeType x_mask, y_mask, z_mask;

    __device__
    explicit cuda_check_intersect_box_t(MortonCodeType query_lower_bound, MortonCodeType query_upper_bound)
        : query_lower_bound(query_lower_bound), query_upper_bound(query_upper_bound) 
    {
        create_masks(x_mask, y_mask, z_mask);
    }

    __device__
    bool operator()(MortonCodeType lower_bound, MortonCodeType upper_bound, const array3d_t<IndexType>& cell_shifts) const override
    {
        return mc_is_less_than_or_equal(query_lower_bound, upper_bound, x_mask, y_mask, z_mask) && 
                mc_is_less_than_or_equal(lower_bound, query_upper_bound, x_mask, y_mask, z_mask);
    }

    __device__
    void operator()(MortonCodeType morton_code, const array3d_t<IndexType>& cell_shifts, IndexType idx, 
        IndexType* indexes, size_t& num_indexes) const override
    {
        if (mc_is_less_than_or_equal(query_lower_bound, morton_code, x_mask, y_mask, z_mask) && 
            mc_is_less_than_or_equal(morton_code, query_upper_bound, x_mask, y_mask, z_mask)) 
        {
            indexes[num_indexes] = idx;
            num_indexes++;  
        }
    }
};

using check_box_t = cuda_check_intersect_box_t<uint32_t, int32_t>;
using result_t = traverse_result_t<int32_t>;

TEST_F(CudaTreeTest, ConstructorTest) {
    vector_host<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 9;

    vector_device<uint32_t> d_morton_codes(morton_codes.size(), stream);
    cudaMemcpyAsync(d_morton_codes.data(), morton_codes.data(), morton_codes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // Create and build the tree
    cuda_binary_radix_tree_t<uint32_t, int32_t> tree = cuda_build_tree(d_morton_codes, num_bits);
    
    // Synchronize before reading results
    cudaStreamSynchronize(stream);
    
    auto pm = gmp::resources::gmp_resource::instance().get_pinned_memory_manager();
    using inode_t = internal_node_t<uint32_t, int32_t>;
    
    // Calculate the number of internal nodes (num_leaf_nodes - 1)
    size_t num_internal_nodes = morton_codes.size() - 1;
    inode_t* internal_nodes = static_cast<inode_t*>(pm->allocate(num_internal_nodes * sizeof(inode_t)));
    tree.get_internal_nodes(internal_nodes);

    // Define expected benchmark values
    std::array<internal_node_t<uint32_t, int32_t>, 4> benchmark = {{
        {8, 4, 0, 511},
        {0, 1, 0, 7},
        {2, 3, 64, 65},
        {6, 7, 0, 127}
    }};

    // Compare actual nodes with benchmark
    for (int i = 0; i < num_internal_nodes; i++) {
        EXPECT_TRUE(compare_nodes(internal_nodes[i], benchmark[i])) 
            << "Node " << i << " mismatch: " << node_to_string(internal_nodes[i]);
    }
    pm->deallocate(internal_nodes);
}

__global__
void traverse_check_box_kernel(cudaTextureObject_t internal_nodes_tex, cudaTextureObject_t leaf_nodes_tex, 
    uint32_t num_leaf_nodes, uint32_t query_lower_bound, uint32_t query_upper_bound, int32_t* indexes, size_t& num_indexes)
{
    num_indexes = 0;
    check_box_t check_method(query_lower_bound, query_upper_bound);
    tree_traverse_kernel(internal_nodes_tex, leaf_nodes_tex, static_cast<int32_t>(num_leaf_nodes), 
        check_method, array3d_t<int32_t>{0, 0, 0}, 
        indexes, num_indexes);
    return;
}

TEST_F(CudaTreeTest, TraverseBasicTest) {
    // Test morton codes
    vector_host<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 3;

    auto dm = gmp::resources::gmp_resource::instance().get_device_memory_manager();
    auto pm = gmp::resources::gmp_resource::instance().get_pinned_memory_manager();
    auto stream = gmp::resources::gmp_resource::instance().get_stream();

    // Create device morton codes
    vector_device<uint32_t> d_morton_codes(morton_codes.size(), stream);
    cudaMemcpyAsync(d_morton_codes.data(), morton_codes.data(), morton_codes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    // Create and build the tree
    cuda_binary_radix_tree_t<uint32_t, int32_t> tree = cuda_build_tree(d_morton_codes, num_bits * 3);
    
    // Synchronize before proceeding
    cudaStreamSynchronize(stream);

    // Define query bounds
    uint32_t query_lower_bound = interleave_bits(0, 0, 0, num_bits);
    uint32_t query_upper_bound = interleave_bits(0b100, 0b100, 0b100, num_bits);

    // Perform traversal
    result_t result(morton_codes.size());

    // Allocate device memory for the counters
    size_t* d_num_indexes = static_cast<size_t*>(dm->allocate(sizeof(size_t), stream));

    traverse_check_box_kernel<<<1, 1, 0, stream>>>(tree.internal_nodes_tex, tree.leaf_nodes_tex, 
        static_cast<uint32_t>(tree.num_leaf_nodes), query_lower_bound, query_upper_bound, 
        result.indexes, *d_num_indexes);

    // Copy results to host
    cudaMemcpyAsync(&result.num_indexes, d_num_indexes, sizeof(size_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Copy results to host for verification
    vector_host<int32_t> h_indexes(result.num_indexes);

    cudaMemcpyAsync(h_indexes.data(), result.indexes, result.num_indexes * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Basic verification
    EXPECT_EQ(result.num_indexes, 3);

    // Cleanup
    dm->deallocate(d_num_indexes, stream);
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
    
    // check_intersect_box_t<uint32_t> op(query_min, query_max);
    check_intersect_box_t<uint32_t, vector<array3d_int32>> op(query_min, query_max);

    // Get results from tree traversal
    auto cpu_result = cpu_tree.traverse(op);

    /////////////////
    // CUDA result //
    /////////////////
    auto dm = gmp::resources::gmp_resource::instance().get_device_memory_manager();
    auto stream = gmp::resources::gmp_resource::instance().get_stream();
    
    vector_device<uint32_t> d_morton_codes(morton_codes.size(), stream);
    cudaMemcpyAsync(d_morton_codes.data(), morton_codes.data(), morton_codes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    cuda_binary_radix_tree_t<uint32_t, int32_t> cuda_tree = cuda_build_tree(d_morton_codes, num_bits * 3);

    result_t cuda_result(morton_codes.size());

    // Allocate device memory for the counters
    size_t* d_num_indexes = static_cast<size_t*>(dm->allocate(sizeof(size_t), stream));
    
    traverse_check_box_kernel<<<1, 1, 0, stream>>>(cuda_tree.internal_nodes_tex, cuda_tree.leaf_nodes_tex, 
        static_cast<uint32_t>(cuda_tree.num_leaf_nodes), query_min, query_max, 
        cuda_result.indexes, *d_num_indexes);
    
    // Copy results to host
    cudaMemcpyAsync(&cuda_result.num_indexes, d_num_indexes, sizeof(size_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    vector_host<int32_t> h_indexes(cuda_result.num_indexes);
    cudaMemcpyAsync(h_indexes.data(), cuda_result.indexes, cuda_result.num_indexes * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // check reuslt 
    EXPECT_EQ(cpu_result.size(), cuda_result.num_indexes);
    std::cout << "result size: " << cpu_result.size() << std::endl;
    for (int i = 0; i < cuda_result.num_indexes; i++) {
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