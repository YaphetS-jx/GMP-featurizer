#include <memory>
#include <chrono>
#include <iostream>
#include "input.hpp"
#include "error.hpp"
#include "atom.hpp"
#include "geometry.hpp"
#include "containers.hpp"
#include "featurizer.hpp"
#include "util.hpp"
#include "cpu_featurizer.hpp"

namespace gmp {

    std::vector<std::vector<gmp::gmp_float>> run_cpu_featurizer(input::input_t* input) 
    {
        using namespace gmp;
        using namespace gmp::containers;
        std::cout << "Running CPU featurizer..." << std::endl;

        // create unit cell
        std::unique_ptr<atom::unit_cell_flt> unit_cell = std::make_unique<atom::unit_cell_flt>(input->files->get_atom_file());
        GMP_CHECK(get_last_error());

        // create psp configuration
        std::unique_ptr<atom::psp_config_flt> psp_config = std::make_unique<atom::psp_config_flt>(input->files->get_psp_file(), unit_cell.get());
        GMP_CHECK(get_last_error());

        // create reference positions
        gmp::containers::vector<gmp::geometry::point3d_t<gmp_float>> ref_positions;
        
        // Check if reference positions are provided directly
        if (input->has_reference_positions()) {
            // Use provided reference positions directly (no copying needed)
            ref_positions = input->get_reference_positions();
        } else if (!input->files->get_reference_grid_file().empty()) {
            // Read reference positions from file
            ref_positions = input->read_reference_grid_from_file(input->files->get_reference_grid_file());
            GMP_CHECK(get_last_error());
        } else {
            // Use original atom::set_ref_positions logic
            auto ref_grid = input->descriptor_config->get_ref_grid();
            ref_positions = atom::set_ref_positions(ref_grid, unit_cell->get_atoms());
        }

        // create featurizer_t
        std::unique_ptr<featurizer::featurizer_flt> featurizer = std::make_unique<featurizer::featurizer_flt>(
            input->descriptor_config.get(), unit_cell.get(), psp_config.get());
        GMP_CHECK(get_last_error());

        // compute features
        auto t1 = std::chrono::high_resolution_clock::now();
        // Convert std::vector to gmp::containers::vector
        gmp::containers::vector<gmp::geometry::point3d_t<gmp_float>> ref_positions_container(ref_positions.begin(), ref_positions.end());
        auto result = featurizer->compute(ref_positions_container, input->descriptor_config.get(), unit_cell.get(), psp_config.get());
        auto t2 = std::chrono::high_resolution_clock::now();
        auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << "CPU featurizer time: " << static_cast<double>(compute_time.count()) << " ms" << std::endl;
        
        return result;
    }

} // namespace gmp
