#include <gtest/gtest.h>
#include "region_query.hpp"
#include "atom.hpp"
#include "morton_codes.hpp"
#include "geometry.hpp"
#include "gmp_float.hpp"
#include <random>

using namespace gmp::region_query;
using namespace gmp::atom;
using namespace gmp::containers;
using namespace gmp::geometry;

TEST(RegionQueryTest, basic_query) {
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
    // auto seed = 2862654988;
    std::mt19937 gen(seed);
    std::cout << "Random seed: " << seed << std::endl;
    std::uniform_real_distribution<gmp::gmp_float> dist(0.0, 1.0);

    // Create random atoms
    std::vector<atom_t<gmp::gmp_float>> atoms;
    auto num_atoms = 1000;
    auto num_bits_per_dim = 5;
    atoms.reserve(num_atoms);

    for (int i = 0; i < num_atoms; i++) {
        gmp::gmp_float x = dist(gen);
        gmp::gmp_float y = dist(gen);
        gmp::gmp_float z = dist(gen);
        atoms.push_back({{x, y, z}, 1.0, 0});
    }
    unit_cell.set_atoms(std::move(atoms));

    // Create region query object
    region_query_t<uint32_t, int32_t, gmp::gmp_float> region_query(&unit_cell, num_bits_per_dim);

    // Test query around a point
    point3d_t<gmp::gmp_float> query_point{0.8071282732743802, 0.7297317866938179, 0.5362280914547007};
    gmp::gmp_float cutoff = 0.9731157639793706;
    std::cout << "query point: " << query_point << std::endl;
    std::cout << "cutoff: " << cutoff << std::endl;

    // build tree
    auto brt = std::make_unique<binary_radix_tree_t<int32_t, gmp::gmp_float>>(region_query.get_unique_morton_codes(), num_bits_per_dim * 3);

    // query
    auto results = region_query.query(query_point, cutoff, brt.get(), &unit_cell);
    std::sort(results.begin(), results.end(), 
    [](const query_result_t<gmp::gmp_float>& a, const query_result_t<gmp::gmp_float>& b) {
        return a.distance_squared < b.distance_squared || 
        (a.distance_squared == b.distance_squared && a.neighbor_index < b.neighbor_index);
    });
    
    std::cout << "query results size: " << results.size() << std::endl;

    // Create compare_op manually for benchmarking
    check_sphere_t<gmp::gmp_float, int32_t> compare_op(num_bits_per_dim, unit_cell.get_periodicity(), unit_cell.get_lattice());
    compare_op.update_point_radius(query_point, cutoff);
    auto cell_shift_start = compare_op.get_cell_shift_start();
    auto cell_shift_end = compare_op.get_cell_shift_end();
    
    std::vector<query_result_t<gmp::gmp_float>> results_benchmark;
    for (auto shift_z = cell_shift_start[2]; shift_z <= cell_shift_end[2]; shift_z++) {
        for (auto shift_y = cell_shift_start[1]; shift_y <= cell_shift_end[1]; shift_y++) {
            for (auto shift_x = cell_shift_start[0]; shift_x <= cell_shift_end[0]; shift_x++) {
                for (auto i = 0; i < num_atoms; i++) {
                    array3d_t<gmp::gmp_float> cell_shift{static_cast<gmp::gmp_float>(shift_x), static_cast<gmp::gmp_float>(shift_y), static_cast<gmp::gmp_float>(shift_z)};
                    array3d_t<gmp::gmp_float> difference;
                    gmp::gmp_float distance2 = unit_cell.get_lattice()->calculate_distance_squared(unit_cell.get_atoms()[i].pos, query_point, cell_shift, difference);
                    if (distance2 < cutoff * cutoff) {
                        array3d_t<gmp::gmp_float> difference_cartesian = unit_cell.get_lattice()->fractional_to_cartesian(difference);
                        results_benchmark.push_back({difference_cartesian, distance2, i});
                    }
                }
            }
        }
    }

    std::sort(results_benchmark.begin(), results_benchmark.end(), 
    [](const query_result_t<gmp::gmp_float>& a, const query_result_t<gmp::gmp_float>& b) {
        return a.distance_squared < b.distance_squared || 
        (a.distance_squared == b.distance_squared && a.neighbor_index < b.neighbor_index);
    });

    EXPECT_EQ(results.size(), results_benchmark.size());
    for (auto i = 0; i < results.size(); i++) {
        EXPECT_EQ(results[i].neighbor_index, results_benchmark[i].neighbor_index);
        EXPECT_DOUBLE_EQ(results[i].distance_squared, results_benchmark[i].distance_squared);
        EXPECT_DOUBLE_EQ(results[i].difference[0], results_benchmark[i].difference[0]);
        EXPECT_DOUBLE_EQ(results[i].difference[1], results_benchmark[i].difference[1]);
        EXPECT_DOUBLE_EQ(results[i].difference[2], results_benchmark[i].difference[2]);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 