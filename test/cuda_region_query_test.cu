#include <gtest/gtest.h>
#include "cuda_region_query.hpp"
#include "region_query.hpp"
#include "atom.hpp"
#include "morton_codes.hpp"
#include "geometry.hpp"
#include <random>

using namespace gmp::region_query;
using namespace gmp::atom;
using namespace gmp::containers;
using namespace gmp::geometry;

class CudaRegionQueryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up any common test data
        stream = gmp::resources::gmp_resource::instance().get_stream();
    }

    void TearDown() override {}
    cudaStream_t stream;
};

TEST_F(CudaRegionQueryTest, BasicQuery) {
    matrix3d_t<double> lattice_vectors = {
        array3d_t<double>{1.0, 0.0, 0.0},
        array3d_t<double>{0.0, 1.0, 0.0},
        array3d_t<double>{0.0, 0.0, 1.0}
    };
    auto lattice = std::make_unique<lattice_t<double>>(lattice_vectors);
    unit_cell_t<double> unit_cell;
    unit_cell.set_lattice(std::move(lattice));
    // Disable periodicity for this test
    unit_cell.set_periodicity(array3d_bool{false, false, false});

    // Create random number generator
    std::random_device rd;
    auto seed = rd();
    std::mt19937 gen(seed);
    std::cout << "Random seed: " << seed << std::endl;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Create random atoms
    vector<atom_t<double>> atoms;
    auto num_atoms = 1000;
    auto num_bits_per_dim = 5;
    atoms.reserve(num_atoms);

    for (int i = 0; i < num_atoms; i++) {
        double x = dist(gen);
        double y = dist(gen);
        double z = dist(gen);
        atoms.emplace_back(x, y, z, 1.0, 0);
    }
    unit_cell.set_atoms(std::move(atoms));

    // Create CPU region query object
    region_query_t<uint32_t, int32_t, double, vector<array3d_int32>> cpu_region_query(&unit_cell, num_bits_per_dim);

    // Create GPU region query object
    cuda_region_query_t<uint32_t, int32_t, double, vector_device<array3d_int32>> gpu_region_query(&unit_cell, num_bits_per_dim, stream);

    // Test query around a point
    point3d_t<double> query_point{0.5, 0.5, 0.5};
    double cutoff = 0.3;
    std::cout << "query point: " << query_point << std::endl;
    std::cout << "cutoff: " << cutoff << std::endl;

    // Get CPU results
    auto cpu_results = cpu_region_query.query(query_point, cutoff, &unit_cell);
    std::sort(cpu_results.begin(), cpu_results.end());

    // Get GPU results (just indices for now)
    auto gpu_results = gpu_region_query.query(query_point, cutoff, &unit_cell, stream);
    cudaStreamSynchronize(stream);

    // Copy GPU results to host for comparison
    vector<int> gpu_results_host(gpu_results.size());
    cudaMemcpyAsync(gpu_results_host.data(), gpu_results.data(), 
        gpu_results.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::sort(gpu_results_host.begin(), gpu_results_host.end());

    // Extract CPU indices for comparison
    vector<int> cpu_indices;
    for (const auto& result : cpu_results) {
        cpu_indices.push_back(result.neighbor_index);
    }
    std::sort(cpu_indices.begin(), cpu_indices.end());

    // Compare results
    EXPECT_EQ(cpu_indices.size(), gpu_results_host.size());
    for (size_t i = 0; i < cpu_indices.size(); i++) {
        EXPECT_EQ(cpu_indices[i], gpu_results_host[i]);
    }

    std::cout << "CPU results size: " << cpu_results.size() << std::endl;
    std::cout << "GPU results size: " << gpu_results_host.size() << std::endl;
}

TEST_F(CudaRegionQueryTest, EmptyQuery) {
    matrix3d_t<double> lattice_vectors = {
        array3d_t<double>{1.0, 0.0, 0.0},
        array3d_t<double>{0.0, 1.0, 0.0},
        array3d_t<double>{0.0, 0.0, 1.0}
    };
    auto lattice = std::make_unique<lattice_t<double>>(lattice_vectors);
    unit_cell_t<double> unit_cell;
    unit_cell.set_lattice(std::move(lattice));

    // Create a few atoms
    vector<atom_t<double>> atoms;
    atoms.emplace_back(0.1, 0.1, 0.1, 1.0, 0);
    atoms.emplace_back(0.9, 0.9, 0.9, 1.0, 0);
    unit_cell.set_atoms(std::move(atoms));

    // Create GPU region query object
    cuda_region_query_t<uint32_t, int32_t, double, vector_device<array3d_int32>> gpu_region_query(&unit_cell, 5, stream);

    // Test query with very small cutoff (should return empty)
    point3d_t<double> query_point{0.5, 0.5, 0.5};
    double cutoff = 0.01;

    auto gpu_results = gpu_region_query.query(query_point, cutoff, &unit_cell, stream);
    cudaStreamSynchronize(stream);

    EXPECT_EQ(gpu_results.size(), 0);
}

TEST_F(CudaRegionQueryTest, LargeQuery) {
    matrix3d_t<double> lattice_vectors = {
        array3d_t<double>{1.0, 0.0, 0.0},
        array3d_t<double>{0.0, 1.0, 0.0},
        array3d_t<double>{0.0, 0.0, 1.0}
    };
    auto lattice = std::make_unique<lattice_t<double>>(lattice_vectors);
    unit_cell_t<double> unit_cell;
    unit_cell.set_lattice(std::move(lattice));
    // Disable periodicity for this test
    unit_cell.set_periodicity(array3d_bool{false, false, false});

    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Create many random atoms
    vector<atom_t<double>> atoms;
    auto num_atoms = 10000;
    auto num_bits_per_dim = 8;
    atoms.reserve(num_atoms);

    for (int i = 0; i < num_atoms; i++) {
        double x = dist(gen);
        double y = dist(gen);
        double z = dist(gen);
        atoms.emplace_back(x, y, z, 1.0, 0);
    }
    unit_cell.set_atoms(std::move(atoms));

    // Create CPU region query object
    region_query_t<uint32_t, int32_t, double, vector<array3d_int32>> cpu_region_query(&unit_cell, num_bits_per_dim);

    // Create GPU region query object
    cuda_region_query_t<uint32_t, int32_t, double, vector_device<array3d_int32>> gpu_region_query(&unit_cell, num_bits_per_dim, stream);

    // Test query with large cutoff (should return many results)
    point3d_t<double> query_point{0.5, 0.5, 0.5};
    double cutoff = 0.8;

    // Get CPU results
    auto cpu_results = cpu_region_query.query(query_point, cutoff, &unit_cell);
    std::sort(cpu_results.begin(), cpu_results.end());

    // Get GPU results (just indices for now)
    auto gpu_results = gpu_region_query.query(query_point, cutoff, &unit_cell, stream);
    cudaStreamSynchronize(stream);

    // Copy GPU results to host for comparison
    vector<int> gpu_results_host(gpu_results.size());
    cudaMemcpyAsync(gpu_results_host.data(), gpu_results.data(), 
        gpu_results.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::sort(gpu_results_host.begin(), gpu_results_host.end());

    // Extract CPU indices for comparison
    vector<int> cpu_indices;
    for (const auto& result : cpu_results) {
        cpu_indices.push_back(result.neighbor_index);
    }
    std::sort(cpu_indices.begin(), cpu_indices.end());

    // Compare results
    EXPECT_EQ(cpu_indices.size(), gpu_results_host.size());
    for (size_t i = 0; i < std::min(cpu_indices.size(), gpu_results_host.size()); i++) {
        EXPECT_EQ(cpu_indices[i], gpu_results_host[i]);
    }

    std::cout << "Large query - CPU results size: " << cpu_results.size() << std::endl;
    std::cout << "Large query - GPU results size: " << gpu_results_host.size() << std::endl;
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 