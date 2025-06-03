#include <gtest/gtest.h>
#include "tree.hpp"
#include "types.hpp"

using namespace gmp::tree;
using namespace gmp::containers;

auto& pool = gmp_resource::instance(64, 1<<20).get_host_memory().get_pool();

// Helper function to compare two internal nodes
bool compare_nodes(const internal_node_t<int32_t>& a, const internal_node_t<int32_t>& b) {
    return a.index == b.index &&
           a.first == b.first &&
           a.last == b.last &&
           a.split == b.split &&
           a.left == b.left &&
           a.right == b.right;
}

// Helper function to print node details for debugging
std::string node_to_string(const internal_node_t<int32_t>& node) {
    std::stringstream ss;
    ss << "Node{index=" << node.index 
       << ", first=" << node.first 
       << ", last=" << node.last 
       << ", split=" << node.split 
       << ", left=" << node.left 
       << ", right=" << node.right << "}";
    return ss.str();
}

TEST(BinaryRadixTreeTest, BasicConstruction) {
    // Test morton codes
    vec<uint32_t> morton_codes = {1, 2, 4, 5, 19, 24, 25, 30};
    
    // Create and build the tree
    binary_radix_tree_t<uint32_t, int32_t> tree;
    tree.build_tree(morton_codes);
    
    // verify all the internal nodes 
    EXPECT_EQ(tree.internal_nodes.size(), 7);

    internal_node_t<int32_t> benchmark[7] = {
        {8, 0, 7, 3, 11, 12},
        {9, 0, 1, 0, 0, 1},
        {10, 2, 3, 2, 2, 3},
        {11, 0, 3, 1, 9, 10},
        {12, 4, 7, 4, 4, 13},
        {13, 5, 7, 6, 14, 7},
        {14, 5, 6, 5, 5, 6}
    };

    for (int i = 0; i < 7; i++) {
        EXPECT_TRUE(compare_nodes(tree.internal_nodes[i], benchmark[i])) << "Node " << i << " mismatch: " << node_to_string(tree.internal_nodes[i]);
    }
}

TEST(BinaryRadixTreeTest, DeltaFunction) {
    vec<uint32_t> morton_codes = {1, 2, 4, 5, 19, 24, 25, 30};
    binary_radix_tree_t<uint32_t, int32_t> tree;
    
    // Test delta between adjacent codes
    EXPECT_EQ(tree.delta(morton_codes, 0, 1), 30); // 1 and 2 differ in last bit
    EXPECT_EQ(tree.delta(morton_codes, 1, 2), 29); // 2 and 4 differ in second to last bit
    EXPECT_EQ(tree.delta(morton_codes, 2, 3), 31); // 4 and 5 differ in last bit
    
    // Test delta between non-adjacent codes
    EXPECT_EQ(tree.delta(morton_codes, 0, 4), 27); // 1 and 19 differ in 4th bit from end
    EXPECT_EQ(tree.delta(morton_codes, 4, 7), 28); // 19 and 30 differ in 3rd bit from end
}
