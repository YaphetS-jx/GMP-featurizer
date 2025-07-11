#pragma once
#include <string>
#include <type_traits>
#include <sstream>
#include <vector>
#include <tuple>
#include <fstream>
#include <iostream>
#include <numeric>      
#include <algorithm>    
#include <thread>
#include <sys/sysinfo.h>

#include "error.hpp"

namespace gmp { namespace util {

    template<typename Container>
    void debug_write_vector(const Container &container, const std::string & filename) {
        // Open the file for writing, truncating it if it exists
        std::ofstream outfile(filename, std::ios::trunc);
        
        // Check if the file was opened successfully
        if (!outfile.is_open()) {
            update_error(error_t::output_file_error);
            return;
        }
        
        // Write the vector data
        for (auto it = container.begin(); it != container.end(); ++it) {
            // Set precision and format (only affects numeric types)
            if constexpr (std::is_floating_point<typename Container::value_type>::value) {
                outfile << std::fixed << std::setprecision(16) << std::setw(12);
            }
            outfile << *it;
            if (std::next(it) != container.end()) {
                outfile << "\n";
            }
        }
        outfile << std::endl;
        
        // Close the file
        outfile.close();
    }

    template<typename Container2D>
    void write_vector_2d(const Container2D &container2D, const std::string & filename) {
        // Open the file for writing, truncating it if it exists
        std::ofstream outfile(filename, std::ios::trunc);
        
        // Check if the file was opened successfully
        if (!outfile.is_open()) {
            update_error(error_t::output_file_error);
            return;
        }
        
        // Write the vector data
        for (auto it = container2D.begin(); it != container2D.end(); ++it) {
            for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
                // Set precision and format (only affects numeric types)
                if constexpr (std::is_floating_point<typename Container2D::value_type::value_type>::value) {
                    outfile << std::fixed << std::setprecision(16) << std::setw(12);
                }
                outfile << *it2;
                if (std::next(it2) != it->end()) {
                    outfile << ", ";
                }
            }
            outfile << "\n";
        }
        outfile << std::endl;
        
        // Close the file
        outfile.close();
    }

    template<typename Container>
    void print_vector(const Container& data, const std::string& name) {
        std::cout << name << ": ";
        for (auto it = data.begin(); it != data.end(); ++it) {
            std::cout << *it << ", ";
        }
        std::cout << std::endl;
    }

    template<typename DataType, typename IndexType, template<typename, typename...> class Container>
    Container<IndexType> sort_indexes(const Container<DataType>& data) 
    {
        Container<IndexType> idx(data.size());
        std::iota(idx.begin(), idx.end(), 0);
        // sort indexes based on comparing values in data using std::stable_sort 
        // instead of std::sort to avoid unnecessary index re-orderings when data 
        // contains elements of equal values 
        std::stable_sort(idx.begin(), idx.end(),
            [&data](size_t i1, size_t i2) {return data[i1] < data[i2];});
        return idx;
    }

    template<typename DataType, typename IndexType, template<typename, typename...> class Container>
    Container<IndexType> sort_indexes(const Container<DataType>& data, int start, int end, int start_idx = 0) 
    {
        Container<IndexType> idx(end - start);
        std::iota(idx.begin(), idx.end(), start_idx);
        // sort indexes based on comparing values in data using std::stable_sort 
        // instead of std::sort to avoid unnecessary index re-orderings when data 
        // contains elements of equal values 
        std::stable_sort(idx.begin(), idx.end(),
            [&data, start](size_t i1, size_t i2) {return data[i1 + start] < data[i2 + start];});
        return idx;
    }

    template<typename DataType, typename IndexType, template<typename, typename...> class Container>
    Container<IndexType> sort_indexes_desc(const Container<DataType>& data, int start, int end, int start_idx = 0) 
    {
        Container<IndexType> idx(end - start);
        std::iota(idx.begin(), idx.end(), start_idx);
        // sort indexes based on comparing values in data using std::stable_sort 
        // instead of std::sort to avoid unnecessary index re-orderings when data 
        // contains elements of equal values 
        std::stable_sort(idx.begin(), idx.end(),
            [&data, start, start_idx](size_t i1, size_t i2) {
                return data[i1 - start_idx + start] > data[i2 - start_idx + start];
            });
        return idx;
    }

    // Get the number of available system threads
    inline size_t get_system_thread_count() {
        // Get the number of available CPU cores
        size_t thread_count = std::thread::hardware_concurrency();
        
        // If hardware_concurrency() returns 0, fall back to sysinfo
        if (thread_count == 0) {
            struct sysinfo si;
            if (sysinfo(&si) == 0) {
                thread_count = si.procs;
            } else {
                // Fallback to a reasonable default
                thread_count = 1;
            }
        }
        
        return thread_count;
    }

}}