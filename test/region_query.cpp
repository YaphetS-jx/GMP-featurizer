#include <gtest/gtest.h>
#include "region_query.hpp"
#include "atom.hpp"
#include "morton_codes.hpp"
#include "geometry.hpp"
#include <random>

using namespace gmp::region_query;
using namespace gmp::atom;
using namespace gmp::containers;
using namespace gmp::geometry;

auto& pool = gmp_resource::instance(64, 1<<25).get_host_memory().get_pool();

TEST(RegionQueryTest, basic_query) {
    matrix3d_flt64 lattice_vectors = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}
    };
    auto lattice = std::make_unique<lattice_t>(lattice_vectors);
    unit_cell_t unit_cell;
    unit_cell.set_lattice(std::move(lattice));

    // Create random number generator
    std::random_device rd;
    auto seed = rd();
    // auto seed = 2862654988;
    std::mt19937 gen(seed);
    std::cout << "Random seed: " << seed << std::endl;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Create random atoms
    vec<atom_t> atoms;
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

    // Create region query object
    region_query_t<uint32_t, int32_t, double, vec<array3d_int32>> region_query(&unit_cell, num_bits_per_dim);

    // Test query around a point
    point_flt64 query_point{0.8071282732743802, 0.7297317866938179, 0.5362280914547007};
    double cutoff = 0.9731157639793706;
    std::cout << "query point: " << query_point << std::endl;
    std::cout << "cutoff: " << cutoff << std::endl;

    auto results = region_query.query(query_point, cutoff, &unit_cell);
    std::sort(results.begin(), results.end());
    std::cout << "query results size: " << results.size() << std::endl;

    auto compare_op = region_query.get_compare_op();
    auto cell_shift_start = compare_op.get_cell_shift_start();
    auto cell_shift_end = compare_op.get_cell_shift_end();
    
    vec<query_result_t<double>> results_benchmark;
    for (auto shift_z = cell_shift_start[2]; shift_z <= cell_shift_end[2]; shift_z++) {
        for (auto shift_y = cell_shift_start[1]; shift_y <= cell_shift_end[1]; shift_y++) {
            for (auto shift_x = cell_shift_start[0]; shift_x <= cell_shift_end[0]; shift_x++) {
                for (auto i = 0; i < num_atoms; i++) {
                    array3d_flt64 cell_shift{static_cast<double>(shift_x), static_cast<double>(shift_y), static_cast<double>(shift_z)};
                    array3d_flt64 difference;
                    double distance2 = unit_cell.get_lattice()->calculate_distance_squared(atoms[i].pos(), query_point, cell_shift, difference);
                    if (distance2 < cutoff * cutoff) {
                        array3d_flt64 difference_cartesian = unit_cell.get_lattice()->fractional_to_cartesian(difference);
                        results_benchmark.emplace_back(difference_cartesian, distance2, i);
                    }
                }
            }
        }
    }

    std::sort(results_benchmark.begin(), results_benchmark.end());

    EXPECT_EQ(results.size(), results_benchmark.size());
    for (auto i = 0; i < results.size(); i++) {
        EXPECT_EQ(results[i].neighbor_index, results_benchmark[i].neighbor_index);
        EXPECT_DOUBLE_EQ(results[i].distance_squared, results_benchmark[i].distance_squared);
        EXPECT_DOUBLE_EQ(results[i].difference[0], results_benchmark[i].difference[0]);
        EXPECT_DOUBLE_EQ(results[i].difference[1], results_benchmark[i].difference[1]);
        EXPECT_DOUBLE_EQ(results[i].difference[2], results_benchmark[i].difference[2]);
    }
}
