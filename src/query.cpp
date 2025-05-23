#include "query.hpp"
#include "util.hpp"

namespace gmp { namespace query {


    query_info_t::query_info_t(const unit_cell_t* unit_cell, const double cutoff) 
        : num_bins_(), bin_atoms_(), bin_offset_(), bin_ranges_()
    {    
        // calculate the interplanar spacing
        array3d_flt64 interplanar_spacing = unit_cell->get_lattice()->get_interplanar_spacing();
        
        // calculate the minimum number of bins
        array3d_flt64 cell_lengths = unit_cell->get_lattice()->get_cell_lengths();

        // calculate the minimum number of bins
        array3d_int32 min_nbins = array3d_int32(
            std::ceil(cell_lengths[0] / cutoff),
            std::ceil(cell_lengths[1] / cutoff),
            std::ceil(cell_lengths[2] / cutoff)
        );

        // calculate the number of bins
        for (auto i = 0; i < 3; i++) {
            num_bins_[i] = static_cast<int> (std::ceil(interplanar_spacing[i] / cutoff)); // Initial guess.
            num_bins_[i] = std::max(num_bins_[i], min_nbins[i]); // Ensure it's at least min_nbins[i].
        }

        // total number of bins
        auto total_nbins = num_bins_.prod();
        
        // allocate the bin atoms and bin offset
        vec<int> bin_num_atoms(total_nbins, 0);
        bin_offset_ = vec<int>(total_nbins + 1, 0);

        // count the number of atoms in each bin 
        auto& atoms = unit_cell->get_atoms();
        for (auto& atom : atoms) {
            auto bin_index = get_bin_index_1d(atom.pos());
            bin_num_atoms[bin_index]++;
        }
        
        // exclusive prefix scan to calculate bin offsets
        bin_offset_[0] = 0;
        for (auto i = 1; i <= total_nbins; i++) {
            bin_offset_[i] = bin_offset_[i - 1] + bin_num_atoms[i - 1];
        }
        assert(bin_offset_.back() == atoms.size());

        // scan for the exact atom index 
        bin_atoms_.resize(atoms.size());
        vec<int> bin_next_write(bin_offset_);
        for (auto i = 0; i < atoms.size(); i++) {
            auto bin_index = get_bin_index_1d(atoms[i].pos());
            bin_atoms_[bin_next_write[bin_index]] = i;
            bin_next_write[bin_index]++;
        }

        // calculate bin ranges         
        for (auto i = 0; i < 3; i++) {
            bin_ranges_[i] = static_cast<int32_t> (std::ceil(cutoff * num_bins_[i] / interplanar_spacing[i]));
        }

    }

    array3d_int32 query_info_t::get_bin_index_3d(const point_flt64& position) const 
    {
        array3d_int32 bin_index;
        bin_index[0] = static_cast<int>(std::floor(position.x / num_bins_[0]));
        if (bin_index[0] == num_bins_[0]) bin_index[0] = num_bins_[0] - 1;
        bin_index[1] = static_cast<int>(std::floor(position.y / num_bins_[1]));
        if (bin_index[1] == num_bins_[1]) bin_index[1] = num_bins_[1] - 1;
        bin_index[2] = static_cast<int>(std::floor(position.z / num_bins_[2]));
        if (bin_index[2] == num_bins_[2]) bin_index[2] = num_bins_[2] - 1;
        return bin_index;
    };

    int query_info_t::get_bin_index_1d(const point_flt64& position) const 
    {
        auto bin_index = get_bin_index_3d(position);
        return bin_index[0] + bin_index[1] * num_bins_[0] + bin_index[2] * num_bins_[0] * num_bins_[1];
    }

    int query_info_t::get_bin_index_1d(const array3d_int32& bin_index) const 
    {
        return bin_index[0] + bin_index[1] * num_bins_[0] + bin_index[2] * num_bins_[0] * num_bins_[1];
    }


    vec<query_result_t> query_info_t::get_neighbor_list(const double cutoff, const point_flt64& position, const unit_cell_t* unit_cell) const
    {
        const double cutoff2 = cutoff * cutoff;        

        // calculate bin index for reference atom
        const array3d_int32 refBinIndex = get_bin_index_3d(position);
        
        // determine boundary for searching        
        const array3d_int32 lowBoundary = { refBinIndex[0] - bin_ranges_[0], refBinIndex[1] - bin_ranges_[1], refBinIndex[2] - bin_ranges_[2] };
        const array3d_int32 upBoundary = { refBinIndex[0] + bin_ranges_[0], refBinIndex[1] + bin_ranges_[1], refBinIndex[2] + bin_ranges_[2] };

        vec<query_result_t> query_results;
        // search for neighbors in the bin range
        for (auto bin_x = lowBoundary[0]; bin_x <= upBoundary[0]; bin_x++) {
            for (auto bin_y = lowBoundary[1]; bin_y <= upBoundary[1]; bin_y++) {
                for (auto bin_z = lowBoundary[2]; bin_z <= upBoundary[2]; bin_z++) {

                    // check if the bin is out of bounds
                    bool out_of_bounds = false;
                    array3d_int32 neighbor_bin_index;
                    array3d_flt64 cell_shift;

                    for (int dim = 0; dim < 3; ++dim) {
                        auto idx = (dim == 0) ? bin_x : (dim == 1) ? bin_y : bin_z;
                        if ((idx < 0 || idx >= num_bins_[dim]) && !unit_cell->get_periodicity()[dim]) {
                            out_of_bounds = true;
                            break;
                        }
                        neighbor_bin_index[dim] = (idx % num_bins_[dim] + num_bins_[dim]) % num_bins_[dim];
                        cell_shift[dim] = static_cast<double>(idx - neighbor_bin_index[dim]) / num_bins_[dim];
                    }
                    if (out_of_bounds) continue;

                    // get the bin index
                    auto bin_index = get_bin_index_1d(neighbor_bin_index);

                    // get the start and end of the bin
                    auto start = bin_offset_[bin_index];
                    auto end = bin_offset_[bin_index + 1];

                    for (auto atom_idx = start; atom_idx < end; atom_idx++) {
                        auto neighbor_index = bin_atoms_[atom_idx];
                        auto neighbor_position = unit_cell->get_atoms()[neighbor_index].pos();

                        // calculate the distance between the reference atom and the neighbor
                        array3d_flt64 difference;
                        auto distance_squared = unit_cell->get_lattice()->calculate_distance_squared(
                            neighbor_position, position, cell_shift, difference);

                        // convert distance to cartesian coordinates
                        array3d_flt64 difference_cartesian = unit_cell->get_lattice()->fractional_to_cartesian(difference);

                        if (distance_squared < cutoff2) {
                            query_results.push_back(query_result_t(difference_cartesian, distance_squared, neighbor_index));
                        }
                    }                    
                }
            }
        }
        return query_results;
    }  

    void query_info_t::dump() const
    {
        std::cout << "num_bins_: " << num_bins_[0] << " " << num_bins_[1] << " " << num_bins_[2] << std::endl;
        util::print_vector(bin_atoms_, "bin_atoms_");
        util::print_vector(bin_offset_, "bin_offset_");
        std::cout << "bin_ranges_: " << bin_ranges_[0] << " " << bin_ranges_[1] << " " << bin_ranges_[2] << std::endl;
    }
}}