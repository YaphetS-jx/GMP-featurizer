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
    std::unique_ptr<atom::unit_cell_t> unit_cell = std::make_unique<atom::unit_cell_t>(input->files->get_atom_file());
    unit_cell->dump();
    GMP_CHECK(get_last_error());

    // create psp configuration
    std::unique_ptr<atom::psp_config_t> psp_config = std::make_unique<atom::psp_config_t>(input->files->get_psp_file(), unit_cell.get());
    psp_config->dump();
    GMP_CHECK(get_last_error());

    // create featurizer_t
    std::unique_ptr<featurizer::featurizer_t> featurizer = std::make_unique<featurizer::featurizer_t>(
        input->descriptor_config.get(), unit_cell.get(), psp_config.get());
    GMP_CHECK(get_last_error());

    // create reference positions
    auto ref_positions = atom::set_ref_positions(input->descriptor_config->get_ref_grid(), unit_cell->get_atoms());
    auto result = featurizer->compute(ref_positions, input->descriptor_config.get(), unit_cell.get(), psp_config.get());
    GMP_CHECK(get_last_error());

    // print result 
    util::write_vector_2d(result, input->files->get_output_file());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken: " << static_cast<double>(duration.count()) / 1000.0 << " seconds" << std::endl;

    return 0;
}
