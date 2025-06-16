#include <gtest/gtest.h>
#include "tree.hpp"
#include "types.hpp"
#include <random>
#include <set>
#include <tuple>

using namespace gmp::tree;
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

TEST(BinaryRadixTreeTest, BasicConstruction) {
    // Test morton codes
    vec<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 9;

    // Create and build the tree
    binary_radix_tree_t<uint32_t, int32_t> tree;
    tree.build_tree(morton_codes, num_bits);
    
    // verify all the internal nodes 
    EXPECT_EQ(tree.get_internal_nodes().size(), 4);

    std::array<internal_node_t<uint32_t, int32_t>, 4> benchmark = {{
        {8, 4, 0, 511},
        {0, 1, 0, 7},
        {2, 3, 64, 65},
        {6, 7, 0, 127}
    }};

    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(compare_nodes(tree.get_internal_nodes()[i], benchmark[i])) << "Node " << i << " mismatch: " << node_to_string(tree.get_internal_nodes()[i]);
    }
}

TEST(BinaryRadixTreeTest, DeltaFunction) {
    vec<uint32_t> morton_codes = {1, 2, 4, 5, 19, 24, 25, 30};
    binary_radix_tree_t<uint32_t, int32_t> tree;
    
    EXPECT_EQ(tree.delta(morton_codes, 0, -1), -1); // out of bounds
    EXPECT_EQ(tree.delta(morton_codes, 0, -1, 10), -1); // out of bounds

    EXPECT_EQ(tree.delta(morton_codes, 3, 5, 32), 27); // 5 and 24 differ 5 bits
    EXPECT_EQ(tree.delta(morton_codes, 3, 5, 10), 5); // 5 and 24 differ 5 bits
}

template <typename MortonCodeType, typename VecType = vec<array3d_int32>>
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

TEST(BinaryRadixTreeTest, Traverse_for_test) {
    vec<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 3;

    // Create and build the tree
    binary_radix_tree_t<uint32_t, int32_t> tree;
    tree.build_tree(morton_codes, num_bits*3);

    uint32_t query_lower_bound = interleave_bits(0, 0, 0, num_bits);
    uint32_t query_upper_bound = interleave_bits(0b100, 0b100, 0b100, num_bits);

    check_intersect_box_t<uint32_t, vec<array3d_int32>> op(query_lower_bound, query_upper_bound);
    auto result = tree.traverse(op);
    EXPECT_EQ(result.size(), 3);
    EXPECT_TRUE(result.find(0) != result.end());
    EXPECT_TRUE(result.find(1) != result.end());
    EXPECT_TRUE(result.find(2) != result.end());
    EXPECT_EQ(result[0].size(), 1);
    EXPECT_EQ(result[1].size(), 1);
    EXPECT_EQ(result[2].size(), 1);
    EXPECT_EQ(result[0][0], (array3d_int32{0, 0, 0}));
    EXPECT_EQ(result[1][0], (array3d_int32{0, 0, 0}));
    EXPECT_EQ(result[2][0], (array3d_int32{0, 0, 0}));
}


TEST(BinaryRadixTreeTest, Traverse_general_case) {
    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15); // 2^4 - 1 = 15

    // Generate random points
    const int num_bits = 4;
    const int max_coord = (1 << num_bits) - 1;
    const int num_points = 5000;

    // Generate points and morton codes
    std::set<std::tuple<int, int, int>> unique_points;
    vec<uint32_t> morton_codes;
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

    // Build tree
    binary_radix_tree_t<uint32_t, int32_t> tree;
    tree.build_tree(morton_codes, num_bits * 3);

    // Generate random query window
    int x1 = dis(gen), x2 = dis(gen);
    int y1 = dis(gen), y2 = dis(gen);
    int z1 = dis(gen), z2 = dis(gen);
    
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    if (z1 > z2) std::swap(z1, z2);

    uint32_t query_min = interleave_bits(x1, y1, z1, num_bits);
    uint32_t query_max = interleave_bits(x2, y2, z2, num_bits);
    // check_intersect_box_t<uint32_t> op(query_min, query_max);
    check_intersect_box_t<uint32_t, vec<array3d_int32>> op(query_min, query_max);

    // Get results from tree traversal
    auto result = tree.traverse(op);

    // get result vec
    vec<int32_t> result_vec;
    for (const auto& [index, shifts] : result) {
        for (const auto& shift : shifts) {
            result_vec.push_back(index);
        }
    }

    // Brute force check
    std::vector<int32_t> bench_result;
    for (size_t i = 0; i < morton_codes.size(); ++i) {
        if (op(morton_codes[i], morton_codes[i])) {
            bench_result.push_back(i);
        }
    }

    // Sort both results
    std::sort(result_vec.begin(), result_vec.end());
    std::sort(bench_result.begin(), bench_result.end());

    std::cout << "query counts: " << result_vec.size() << std::endl;

    // Compare results
    EXPECT_EQ(result.size(), bench_result.size()) << "Result sizes don't match!";
    EXPECT_TRUE(std::equal(result_vec.begin(), result_vec.end(), bench_result.begin())) 
        << "Results don't match!";
}

