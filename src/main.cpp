#include "input.hpp"
#include "error.hpp"
#include "atom.hpp"
#include "geometry.hpp"
#include "types.hpp"
#include "featurizer.hpp"
#include "util.hpp"
#include <chrono>

int main(int argc, char* argv[]) {
    using namespace gmp;
    using namespace gmp::containers;

    if (argc != 2) {
        std::cout << "ERROR: Please provide the path to the JSON file." << std::endl;
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    // parse arguments
    std::unique_ptr<input::input_t> input = std::make_unique<input::input_t>(argv[1]);
    GMP_CHECK(get_last_error());    

    // create unit cell
    std::unique_ptr<atom::unit_cell_flt64> unit_cell = std::make_unique<atom::unit_cell_flt64>(input->files->get_atom_file());
    GMP_CHECK(get_last_error());

    // create psp configuration
    std::unique_ptr<atom::psp_config_flt64> psp_config = std::make_unique<atom::psp_config_flt64>(input->files->get_psp_file(), unit_cell.get());
    GMP_CHECK(get_last_error());

    // create reference positions
    auto ref_positions = atom::set_ref_positions(input->descriptor_config->get_ref_grid(), unit_cell->get_atoms());

    // create featurizer_t
    std::unique_ptr<featurizer::featurizer_flt64> featurizer = std::make_unique<featurizer::featurizer_flt64>(
        input->descriptor_config.get(), unit_cell.get(), psp_config.get());
    GMP_CHECK(get_last_error());

    // compute features
    auto t1 = std::chrono::high_resolution_clock::now();
    auto result = featurizer->compute(ref_positions, input->descriptor_config.get(), unit_cell.get(), psp_config.get());
    auto t2 = std::chrono::high_resolution_clock::now();
    auto compute_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "compute time: " << static_cast<double>(compute_time.count()) / 1000.0 << " seconds" << std::endl;
    util::write_vector_2d(result, input->files->get_output_file());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto walltime = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken: " << static_cast<double>(walltime.count()) / 1000.0 << " seconds" << std::endl;

    return 0;
}
