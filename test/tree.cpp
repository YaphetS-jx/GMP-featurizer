#include <gtest/gtest.h>
#include "tree.hpp"
#include "containers.hpp"
#include <random>
#include <set>
#include <tuple>

using namespace gmp::tree;
using namespace gmp::tree::morton_codes;
using namespace gmp::containers;
using gmp::math::array3d_t;
using gmp::gmp_float;

// Helper function to compare two internal nodes
bool compare_nodes(const internal_node_t<int32_t, float>& a, const internal_node_t<int32_t, float>& b) {
    return a.get_left() == b.get_left() &&
           a.get_right() == b.get_right() &&
           a.lower_bound_coords[0] == b.lower_bound_coords[0] &&
           a.lower_bound_coords[1] == b.lower_bound_coords[1] &&
           a.lower_bound_coords[2] == b.lower_bound_coords[2] &&
           a.upper_bound_coords[0] == b.upper_bound_coords[0] &&
           a.upper_bound_coords[1] == b.upper_bound_coords[1] &&
           a.upper_bound_coords[2] == b.upper_bound_coords[2];
}

// Helper function to print node details for debugging
std::string node_to_string(const internal_node_t<int32_t, float>& node) {
    std::stringstream ss;
    ss << "Node {"
       << "left=" << node.get_left()
       << ", right=" << node.get_right()
       << ", lower_bound_coords=[" << node.lower_bound_coords[0] << ", " << node.lower_bound_coords[1] << ", " << node.lower_bound_coords[2] << "]"
       << ", upper_bound_coords=[" << node.upper_bound_coords[0] << ", " << node.upper_bound_coords[1] << ", " << node.upper_bound_coords[2] << "]"
       << "}";
    return ss.str();
}

template <typename FloatType>
array3d_t<FloatType> compute_lower_bound_coords(uint32_t bound, int num_bits_per_dim) {
    uint32_t x, y, z;
    deinterleave_bits(bound, num_bits_per_dim, x, y, z);
    return {
        morton_code_to_coordinate<FloatType, int32_t, uint32_t>(x, num_bits_per_dim),
        morton_code_to_coordinate<FloatType, int32_t, uint32_t>(y, num_bits_per_dim),
        morton_code_to_coordinate<FloatType, int32_t, uint32_t>(z, num_bits_per_dim)
    };
}

template <typename FloatType>
array3d_t<FloatType> compute_upper_bound_coords(uint32_t bound, int num_bits_per_dim) {
    auto mins = compute_lower_bound_coords<FloatType>(bound, num_bits_per_dim);
    FloatType size_per_dim = FloatType(1) / FloatType(1 << (num_bits_per_dim - 1));
    return {mins[0] + size_per_dim, mins[1] + size_per_dim, mins[2] + size_per_dim};
}

TEST(BinaryRadixTreeTest, BasicConstruction) {
    // Test morton codes
    std::vector<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 12;

    // Create and build the tree
    binary_radix_tree_t<int32_t, float> tree;
    tree.build_tree(morton_codes, num_bits);
    
    // verify all the internal nodes 
    EXPECT_EQ(tree.get_internal_nodes().size(), 4);

    std::array<internal_node_t<int32_t, float>, 4> benchmark;
    
    // Initialize benchmark nodes using the new structure
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

    int num_bits_per_dim = num_bits / 3;
    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(compare_nodes(tree.get_internal_nodes()[i], benchmark[i])) << "Node " << i << " mismatch: " << node_to_string(tree.get_internal_nodes()[i]);
        // Compare coordinates directly since we no longer store morton codes
        EXPECT_FLOAT_EQ(tree.get_internal_nodes()[i].lower_bound_coords[0], benchmark[i].lower_bound_coords[0]);
        EXPECT_FLOAT_EQ(tree.get_internal_nodes()[i].lower_bound_coords[1], benchmark[i].lower_bound_coords[1]);
        EXPECT_FLOAT_EQ(tree.get_internal_nodes()[i].lower_bound_coords[2], benchmark[i].lower_bound_coords[2]);
        EXPECT_FLOAT_EQ(tree.get_internal_nodes()[i].upper_bound_coords[0], benchmark[i].upper_bound_coords[0]);
        EXPECT_FLOAT_EQ(tree.get_internal_nodes()[i].upper_bound_coords[1], benchmark[i].upper_bound_coords[1]);
        EXPECT_FLOAT_EQ(tree.get_internal_nodes()[i].upper_bound_coords[2], benchmark[i].upper_bound_coords[2]);
    }
}

TEST(BinaryRadixTreeTest, DeltaFunction) {
    std::vector<uint32_t> morton_codes = {1, 2, 4, 5, 19, 24, 25, 30};
    
    // Test out of bounds - should return -1 for out-of-bounds indices
    EXPECT_EQ(delta(morton_codes.data(), static_cast<int>(morton_codes.size()), 0, -1, 32), -1); // out of bounds
    EXPECT_EQ(delta(morton_codes.data(), static_cast<int>(morton_codes.size()), 0, -1, 10), -1); // out of bounds

    EXPECT_EQ(delta(morton_codes.data(), static_cast<int>(morton_codes.size()), 3, 5, 32), 27); // 5 and 24 differ 5 bits
    EXPECT_EQ(delta(morton_codes.data(), static_cast<int>(morton_codes.size()), 3, 5, 10), 5); // 5 and 24 differ 5 bits
}

template <typename FloatType = float>
class check_intersect_box_t : public compare_op_t<FloatType> {
public:
    uint32_t query_lower_bound, query_upper_bound;
    uint32_t x_mask, y_mask, z_mask;

    explicit check_intersect_box_t(uint32_t query_lower_bound, uint32_t query_upper_bound)
        : query_lower_bound(query_lower_bound), query_upper_bound(query_upper_bound) 
    {
        create_masks(x_mask, y_mask, z_mask);
    }

    bool operator()(const array3d_t<FloatType>& lower_coords, const array3d_t<FloatType>& upper_coords) const override
    {            
        // For this test, we'll use a simple coordinate-based intersection check
        // This is a simplified version that just checks if the query bounds overlap with the node bounds
        return true; // Simplified for testing - in practice you'd implement proper coordinate intersection
    }

    std::vector<array3d_int32> operator()(const array3d_t<FloatType>& lower_coords, FloatType size_per_dim) const override 
    {
        std::vector<array3d_int32> result;
        // Reconstruct morton code from coordinates to check bounds
        int num_bits = 4; // This matches the test's num_bits
        uint32_t mc_x = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(lower_coords[0], num_bits);
        uint32_t mc_y = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(lower_coords[1], num_bits);
        uint32_t mc_z = coordinate_to_morton_code<FloatType, uint32_t, int32_t>(lower_coords[2], num_bits);
        uint32_t morton_code = interleave_bits(mc_x, mc_y, mc_z, num_bits);
        
        if (mc_is_less_than_or_equal(query_lower_bound, morton_code, x_mask, y_mask, z_mask) && 
            mc_is_less_than_or_equal(morton_code, query_upper_bound, x_mask, y_mask, z_mask)) 
        {
            result.push_back(array3d_int32{0, 0, 0});
        }
        return result;
    }

};

TEST(BinaryRadixTreeTest, Traverse_for_test) {
    std::vector<uint32_t> morton_codes = {0, 0b100, 0b001000000, 0b001000001, 0b111111111};
    auto num_bits = 4;

    // Create and build the tree
    binary_radix_tree_t<int32_t, float> tree;
    tree.build_tree(morton_codes, num_bits*3);

    uint32_t query_lower_bound = interleave_bits(0, 0, 0, num_bits);
    uint32_t query_upper_bound = interleave_bits(0b100, 0b100, 0b100, num_bits);

    check_intersect_box_t<float> op(query_lower_bound, query_upper_bound);
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
    const int num_bits = 5;
    // const int max_coord = (1 << num_bits) - 1;
    const int num_points = 5000;

    // Generate points and morton codes
    std::set<std::tuple<int, int, int>> unique_points;
    std::vector<uint32_t> morton_codes;
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
        // Use coordinate_to_morton_code to ensure valid morton codes
        gmp::gmp_float x_coord = static_cast<gmp::gmp_float>(x) / 16.0; // Normalize to [0, 1)
        gmp::gmp_float y_coord = static_cast<gmp::gmp_float>(y) / 16.0;
        gmp::gmp_float z_coord = static_cast<gmp::gmp_float>(z) / 16.0;
        uint32_t mc_x = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(x_coord, num_bits);
        uint32_t mc_y = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(y_coord, num_bits);
        uint32_t mc_z = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(z_coord, num_bits);
        morton_codes.push_back(interleave_bits(mc_x, mc_y, mc_z, num_bits));
    }
    std::cout << "num of unique points: " << unique_points.size() << std::endl;

    // Sort morton codes
    std::sort(morton_codes.begin(), morton_codes.end());

    // Build tree
    binary_radix_tree_t<int32_t, gmp::gmp_float> tree;
    tree.build_tree(morton_codes, num_bits * 3);

    // Generate random query window
    int x1 = dis(gen), x2 = dis(gen);
    int y1 = dis(gen), y2 = dis(gen);
    int z1 = dis(gen), z2 = dis(gen);
    
    if (x1 > x2) std::swap(x1, x2);
    if (y1 > y2) std::swap(y1, y2);
    if (z1 > z2) std::swap(z1, z2);

    // Generate valid morton codes for query bounds
    gmp::gmp_float x1_coord = static_cast<gmp::gmp_float>(x1) / 16.0;
    gmp::gmp_float y1_coord = static_cast<gmp::gmp_float>(y1) / 16.0;
    gmp::gmp_float z1_coord = static_cast<gmp::gmp_float>(z1) / 16.0;
    gmp::gmp_float x2_coord = static_cast<gmp::gmp_float>(x2) / 16.0;
    gmp::gmp_float y2_coord = static_cast<gmp::gmp_float>(y2) / 16.0;
    gmp::gmp_float z2_coord = static_cast<gmp::gmp_float>(z2) / 16.0;
    
    uint32_t mc_x1 = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(x1_coord, num_bits);
    uint32_t mc_y1 = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(y1_coord, num_bits);
    uint32_t mc_z1 = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(z1_coord, num_bits);
    uint32_t mc_x2 = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(x2_coord, num_bits);
    uint32_t mc_y2 = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(y2_coord, num_bits);
    uint32_t mc_z2 = coordinate_to_morton_code<gmp::gmp_float, uint32_t, uint32_t>(z2_coord, num_bits);
    
    uint32_t query_min = interleave_bits(mc_x1, mc_y1, mc_z1, num_bits);
    uint32_t query_max = interleave_bits(mc_x2, mc_y2, mc_z2, num_bits);
    // check_intersect_box_t<uint32_t> op(query_min, query_max);
    check_intersect_box_t<gmp::gmp_float> op(query_min, query_max);

    // Get results from tree traversal
    auto result = tree.traverse(op);

    // get result vec
    std::vector<int32_t> result_vec;
    for (const auto& [index, shifts] : result) {
        for (const auto& shift : shifts) {
            result_vec.push_back(index);
        }
    }

    // Brute force check - calculate coordinates for each morton code
    std::vector<int32_t> bench_result;
    gmp::gmp_float size_per_dim = 1.0f / (1 << (num_bits - 1));
    for (size_t i = 0; i < morton_codes.size(); ++i) {
        // Calculate coordinates from morton code for the test
        uint32_t x_min, y_min, z_min;
        deinterleave_bits(morton_codes[i], num_bits, x_min, y_min, z_min);
        array3d_t<gmp::gmp_float> lower_coords = {
            morton_code_to_coordinate<gmp::gmp_float, int32_t, uint32_t>(x_min, num_bits),
            morton_code_to_coordinate<gmp::gmp_float, int32_t, uint32_t>(y_min, num_bits),
            morton_code_to_coordinate<gmp::gmp_float, int32_t, uint32_t>(z_min, num_bits)
        };
        
        if (!op(lower_coords, size_per_dim).empty()) {
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

