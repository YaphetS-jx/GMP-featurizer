#include <gtest/gtest.h>
#include "containers.hpp"
#include "resources.hpp"

using namespace gmp::containers;

TEST(gmp_resources, custom_containers) {
    std::cout << "--------------------------------" << std::endl;
    std::cout << "GMPResourcesTest, CustomContainers" << std::endl;
    std::cout << "--------------------------------" << std::endl;    

    // Test vec with custom pool
    {
        std::vector<int> v;
        v.push_back(1);
        v.push_back(2);
        v.push_back(3);
        EXPECT_EQ(v.size(), 3);
        EXPECT_EQ(v[0], 1);
        EXPECT_EQ(v[1], 2);
        EXPECT_EQ(v[2], 3);
    }

    // Test deque with default pool
    {
        deque<int> d;
        d.push_back(1);
        d.push_front(2);
        d.push_back(3);
        EXPECT_EQ(d.size(), 3);
        EXPECT_EQ(d[0], 2);
        EXPECT_EQ(d[1], 1);
        EXPECT_EQ(d[2], 3);
    }

    // Test deque with custom pool
    {
        deque<int> d;
        d.push_back(1);
        d.push_front(2);
        d.push_back(3);
        EXPECT_EQ(d.size(), 3);
        EXPECT_EQ(d[0], 2);
        EXPECT_EQ(d[1], 1);
        EXPECT_EQ(d[2], 3);
    }

    // Test stack with default pool
    {
        stack<int> s;
        s.push(1);
        s.push(2);
        s.push(3);
        EXPECT_EQ(s.size(), 3);
        EXPECT_EQ(s.top(), 3);
        s.pop();
        EXPECT_EQ(s.top(), 2);
        s.pop();
        EXPECT_EQ(s.top(), 1);
        s.pop();
        EXPECT_TRUE(s.empty());
    }

    // Test stack with custom pool
    {
        stack<int> s;
        s.push(1);
        s.push(2);
        s.push(3);
        EXPECT_EQ(s.size(), 3);
        EXPECT_EQ(s.top(), 3);
        s.pop();
        EXPECT_EQ(s.top(), 2);
        s.pop();
        EXPECT_EQ(s.top(), 1);
        s.pop();
        EXPECT_TRUE(s.empty());
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 