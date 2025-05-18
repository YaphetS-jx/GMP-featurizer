#include <gtest/gtest.h>
#include "resources.hpp"
#include "boost_pool.hpp"
#include "atom.hpp"
#include "input.hpp"
#include "types.hpp"
using namespace gmp::resources;
using namespace gmp::atom;

// Test singleton instance
TEST(gmp_resources, singleton) {
    std::cout << "--------------------------------" << std::endl;
    std::cout << "GMPResourcesTest, SingletonInstance" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    using namespace gmp::containers;
    // Get the resource instance and initialize host memory pool
    auto& resources = gmp_resource::instance(128, 1<<20);
    auto& host_memory = resources.get_host_memory();

    auto& resources2 = gmp_resource::instance(64, 1<<10);
    EXPECT_EQ(&resources, &resources2) << "Singleton instances should be the same";
    auto& host_memory2 = HostMemory::instance();
    EXPECT_EQ(&host_memory, &host_memory2) << "Singleton instances should be the same";

    // Create a vector using the custom allocator
    using IntVec = vec<int>;  // Using our vec type alias
    IntVec vec(host_memory.get_allocator<int>());

    // Test the vector
    vec.push_back(42);
    vec.push_back(123);
    std::cout << "Vector size: " << vec.size() << std::endl;
    std::cout << "Vector elements: " << vec[0] << ", " << vec[1] << std::endl;

    // Print memory usage information
    host_memory.print_memory_info();
}

class test_t {
public:
    test_t(int n) : n(n) {
        std::cout << "test_t constructor" << std::endl;
    }
    ~test_t() {
        std::cout << "test_t destructor" << std::endl;
    }
    int n;
};

TEST(gmp_resources, smart_pointers) {
    std::cout << "--------------------------------" << std::endl;
    std::cout << "GMPResourcesTest, SmartPointers" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    using namespace gmp::resources;

    auto& pool = gmp_resource::instance(128, 1<<20).get_host_memory().get_pool();
    auto test = make_pool_unique<test_t>(pool, 1);
    
    // Verify the object was created correctly
    EXPECT_EQ(test->n, 1);

    // Create a shared pointer and multiple references to it
    auto test2 = make_pool_shared<test_t>(pool, 2);
    EXPECT_EQ(test2->n, 2);
    
    // Create additional references
    auto test2_ref1 = test2;  // Reference count = 2
    auto test2_ref2 = test2;  // Reference count = 3
    auto test2_ref3 = test2;  // Reference count = 4
    
    std::cout << "Initial memory state:" << std::endl;
    gmp_resource::instance().get_host_memory().print_memory_info();
    
    // Destroy references one by one
    std::cout << "Destroying first reference" << std::endl;
    test2_ref1.reset();
    
    std::cout << "Destroying second reference" << std::endl; 
    test2_ref2.reset();
    
    std::cout << "Destroying third reference" << std::endl;
    test2_ref3.reset();
    
    std::cout << "Destroying final reference" << std::endl;
    test2.reset(); 
    // dtor should be called here
}

TEST(gmp_resources, class_sizes) {
    std::cout << "--------------------------------" << std::endl;
    std::cout << "GMPResourcesTest, ClassSizes" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    using namespace gmp::atom;
    using namespace gmp::geometry;
    using namespace gmp::input;
    auto& pool = gmp_resource::instance(128, 1<<20).get_host_memory().get_pool();

    // Print sizes of atom system related classes    
    std::cout << "Size of atom_t: " << sizeof(atom_t) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(atom_t) <= pool.get_requested_size());
    std::cout << "Size of unit_cell_t: " << sizeof(unit_cell_t) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(unit_cell_t) <= pool.get_requested_size());
    std::cout << "Size of point_flt64: " << sizeof(point_flt64) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(point_flt64) <= pool.get_requested_size());
    
    // Print sizes of geometry related classes
    std::cout << "Size of lattice_t: " << sizeof(lattice_t) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(lattice_t) <= pool.get_requested_size());
    std::cout << "Size of array3d_flt64: " << sizeof(array3d_flt64) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(array3d_flt64) <= pool.get_requested_size());
    std::cout << "Size of matrix3d_flt64: " << sizeof(matrix3d_flt64) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(matrix3d_flt64) <= pool.get_requested_size());
    std::cout << "Size of sym_matrix3d_flt64: " << sizeof(sym_matrix3d_flt64) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(sym_matrix3d_flt64) <= pool.get_requested_size());

    // Print sizes of input related classes
    std::cout << "Size of input_t: " << sizeof(input_t) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(input_t) <= pool.get_requested_size());
    std::cout << "Size of file_path_t: " << sizeof(file_path_t) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(file_path_t) <= pool.get_requested_size());
    std::cout << "Size of feature_t: " << sizeof(feature_t) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(feature_t) <= pool.get_requested_size());
    std::cout << "Size of descriptor_config_t: " << sizeof(descriptor_config_t) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(descriptor_config_t) <= pool.get_requested_size());
    std::cout << "Size of reference_config_t: " << sizeof(reference_config_t) << " bytes" << std::endl;
    EXPECT_TRUE(sizeof(reference_config_t) <= pool.get_requested_size());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 