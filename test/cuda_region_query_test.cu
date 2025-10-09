#include <gtest/gtest.h>
#include "region_query.hpp"
#include "atom.hpp"
#include "morton_codes.hpp"
#include "geometry.hpp"
#include "cuda_region_query.hpp"
#include "cuda_tree.hpp"
#include "gmp_float.hpp"
#include <random>

using namespace gmp::region_query;
using namespace gmp::atom;
using namespace gmp::containers;
using namespace gmp::geometry;
using namespace gmp::tree;

TEST(CudaRegionQueryTest, basic_query) {
    auto stream = gmp::resources::gmp_resource::instance().get_stream();

    // create lattice using gmp_float for precision consistency
    matrix3d_t<gmp::gmp_float> lattice_vectors = {
        array3d_t<gmp::gmp_float>{1.0, 0.0, 0.0},
        array3d_t<gmp::gmp_float>{0.0, 1.0, 0.0},
        array3d_t<gmp::gmp_float>{0.0, 0.0, 1.0}
    };
    auto lattice = std::make_unique<lattice_t<gmp::gmp_float>>(lattice_vectors);
    unit_cell_t<gmp::gmp_float> unit_cell;
    unit_cell.set_lattice(std::move(lattice));

    // Create random number generator
    std::random_device rd;
    auto seed = rd();
    // auto seed = 1772410108;
    std::mt19937 gen(seed);
    std::cout << "Random seed: " << seed << std::endl;
    std::uniform_real_distribution<gmp::gmp_float> dist(0.0, 1.0);

    // Create random atoms
    vector<atom_t<gmp::gmp_float>> atoms;
    auto num_atoms = 1000;
    auto num_bits_per_dim = 3;
    atoms.reserve(num_atoms);

    for (int i = 0; i < num_atoms; i++) {
        gmp::gmp_float x = dist(gen);
        gmp::gmp_float y = dist(gen);
        gmp::gmp_float z = dist(gen);
        atoms.push_back({{x, y, z}, 1.0, 0});
    }

    vector_device<atom_t<gmp::gmp_float>> d_atoms(num_atoms, stream);
    cudaMemcpyAsync(d_atoms.data(), atoms.data(), num_atoms * sizeof(atom_t<gmp::gmp_float>), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    unit_cell.set_atoms(std::move(atoms));

    // Test query around a point
    vector<point3d_t<gmp::gmp_float>> query_points = {
        {dist(gen), dist(gen), dist(gen)},
        {dist(gen), dist(gen), dist(gen)}, 
        {dist(gen), dist(gen), dist(gen)}
    };
    std::cout << "num query points: " << query_points.size() << std::endl;
    for (const auto& query_point : query_points) {
        std::cout << "query point: " << query_point << std::endl;
    }
    gmp::gmp_float cutoff = dist(gen);
    std::cout << "cutoff: " << cutoff << std::endl;

    vector_device<point3d_t<gmp::gmp_float>> d_query_points(query_points.size(), stream);
    cudaMemcpyAsync(d_query_points.data(), query_points.data(), query_points.size() * sizeof(point3d_t<gmp::gmp_float>), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    //////////////////
    //// CPU test ////
    //////////////////
    // Create region query object  
    region_query_t<uint32_t, int32_t, gmp::gmp_float> region_query(&unit_cell, num_bits_per_dim);

    // build tree
    auto brt = std::make_unique<binary_radix_tree_t<int32_t, gmp::gmp_float>>(region_query.get_unique_morton_codes(), num_bits_per_dim * 3);

    // query
    std::vector<std::vector<query_result_t<gmp::gmp_float>>> cpu_results;
    for (const auto& query_point : query_points) {
        std::vector<query_result_t<gmp::gmp_float>> results = region_query.query(query_point, cutoff, brt.get(), &unit_cell);
        cpu_results.push_back(std::move(results));
    }

    int total_size_cpu = 0; 
    vector<int> cpu_results_size;
    for (auto& results : cpu_results) {
        total_size_cpu += results.size();
        cpu_results_size.push_back(total_size_cpu);
    }
    // std::cout << "total cpu query results size: " << total_size_cpu << std::endl;

    //////////////////
    //// GPU test ////
    //////////////////
    // create cuda region query object
    cuda_region_query_t<uint32_t, int32_t, gmp::gmp_float> d_region_query(
        region_query.get_unique_morton_codes(), num_bits_per_dim, 
        region_query.get_offsets(), region_query.get_sorted_indexes());

    vector_device<cuda_query_result_t<gmp::gmp_float>> query_results(1, stream);
    vector_device<int32_t> query_offsets(1, stream);
    cuda_region_query(d_query_points, cutoff, d_region_query, 
        unit_cell.get_lattice(), d_atoms, query_results, query_offsets, stream);

    //////////////////////
    //// Check result ////
    //////////////////////
    vector<int32_t> h_query_offsets(query_offsets.size());
    cudaMemcpyAsync(h_query_offsets.data(), query_offsets.data(), query_offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    EXPECT_EQ(cpu_results_size.size(), query_points.size());

    for (auto i = 0; i < h_query_offsets.size(); i++) {
        EXPECT_EQ(h_query_offsets[i], cpu_results_size[i]);
    }

    vector<cuda_query_result_t<gmp::gmp_float>> h_query_results(query_results.size());
    cudaMemcpyAsync(h_query_results.data(), query_results.data(), query_results.size() * sizeof(cuda_query_result_t<gmp::gmp_float>), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::cout << "query results size: " << h_query_results.size() << std::endl;
    EXPECT_EQ(h_query_results.size(), total_size_cpu);

    // check result 
    for (auto i = 0; i < query_points.size(); i++) {
        int gpu_offset = i == 0 ? 0 : h_query_offsets[i - 1];
        for (auto j = 0; j < cpu_results[i].size(); j++) {
            EXPECT_FLOAT_EQ(h_query_results[gpu_offset + j].distance_squared, cpu_results[i][j].distance_squared);
            EXPECT_EQ(h_query_results[gpu_offset + j].neighbor_index, cpu_results[i][j].neighbor_index);
            EXPECT_EQ(h_query_results[gpu_offset + j].difference[0], cpu_results[i][j].difference[0]);
            EXPECT_EQ(h_query_results[gpu_offset + j].difference[1], cpu_results[i][j].difference[1]);
            EXPECT_EQ(h_query_results[gpu_offset + j].difference[2], cpu_results[i][j].difference[2]);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    gmp::resources::gmp_resource::instance().cleanup();
    return result;
}