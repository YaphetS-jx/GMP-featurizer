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

#include "error.hpp"

namespace gmp { namespace util {

    // Helper function to trim whitespace from a string
    inline std::string trim(const std::string& str) {
        const auto start = str.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        
        const auto end = str.find_last_not_of(" \t\r\n");
        return str.substr(start, end - start + 1);
    }

    // Helper to check if type can be read from stringstream using SFINAE
    template<typename T, typename = void>
    struct is_streamable : std::false_type {};

    template<typename T>
    struct is_streamable<T, std::void_t<decltype(std::declval<std::stringstream&>() >> std::declval<T&>())>> 
        : std::true_type {};

    // Base case for recursion
    template<typename T>
    void parse_values_impl(std::stringstream& ss, T& value) {
        std::string token;
        if (std::getline(ss, token, ',')) {
            std::stringstream value_ss(trim(token));
            if constexpr (is_streamable<T>::value) {
                value_ss >> value;
            } else {
                update_error(error_t::unstreamable_string);
                return;
            }
        }
    }

    // Recursive case for parsing multiple values
    template<typename T, typename... Rest>
    void parse_values_impl(std::stringstream& ss, T& value, Rest&... rest) {
        std::string token;
        if (std::getline(ss, token, ',')) {
            std::stringstream value_ss(trim(token));
            if constexpr (is_streamable<T>::value) {
                value_ss >> value;
            } else {
                update_error(error_t::unstreamable_string);
                return;
            }
        }
        parse_values_impl(ss, rest...);
    }

    // Main parsing function that takes a line and references to variables
    template<typename... Types>
    void parse_line(const std::string& line, Types&... values) {
        std::stringstream ss(line);
        parse_values_impl(ss, values...);
    }

    // Parse single value of type T
    template<typename T>
    bool parse_single_value(std::stringstream& ss, T& value) {
        std::string token;
        if (!std::getline(ss, token, ',')) {
            if (ss.eof()) {
                token = ss.str();
            } else {
                return false;
            }
        }
        
        token = trim(token);
        if (token.empty()) {
            return false;
        }
        
        std::stringstream value_ss(token);
        if constexpr (is_streamable<T>::value) {
            value_ss >> value;
            if (value_ss.fail()) {
                update_error(error_t::unstreamable_string);
                return false;
            }
            return true;
        } else {
            update_error(error_t::unstreamable_string);
            return false;
        }
    }

    namespace detail {
        template<typename T>
        struct is_tuple : std::false_type {};

        template<typename... Args>
        struct is_tuple<std::tuple<Args...>> : std::true_type {};

        // Count non-empty values in a comma-separated string
        inline size_t count_values(const std::string& str) {
            if (str.empty()) return 0;
            
            size_t count = 1;
            bool last_was_comma = false;
            
            for (char c : str) {
                if (c == ',') {
                    count++;
                    last_was_comma = true;
                } else if (!std::isspace(c)) {
                    last_was_comma = false;
                }
            }
            
            // If the string ends with a comma, reduce count
            if (last_was_comma) count--;
            
            return count;
        }
    }

    // Parse a line into a vector of values
    template<typename... Types>
    auto parse_line_pattern(const std::string& line) {
        update_error(error_t::success); // Reset error state at start
        
        if (line.empty()) {
            return std::vector<typename std::conditional<
                sizeof...(Types) == 1,
                std::tuple_element_t<0, std::tuple<Types...>>,
                std::tuple<Types...>
            >::type>();
        }
        
        // For multi-type parsing, validate that we have complete sets
        if constexpr (sizeof...(Types) > 1) {
            size_t value_count = detail::count_values(line);
            if (value_count % sizeof...(Types) != 0) {
                update_error(error_t::incomplete_data_set);
                return std::vector<std::tuple<Types...>>();
            }
        }
        
        if constexpr (sizeof...(Types) == 1) {
            using T = std::tuple_element_t<0, std::tuple<Types...>>;
            std::vector<T> result;
            std::stringstream ss(line);
            T value;
            
            while (true) {
                if (parse_single_value(ss, value)) {
                    result.push_back(value);
                } else if (get_last_error() != error_t::success) {
                    result.clear();
                    return result;
                }
                
                // Check for more data
                std::string rest;
                std::getline(ss, rest);
                rest = trim(rest);
                if (rest.empty()) {
                    break;
                }
                ss.str(rest);
                ss.clear();
            }
            
            return result;
        } else {
            std::vector<std::tuple<Types...>> result;
            std::string remaining = line;
            
            while (!remaining.empty()) {
                std::stringstream ss(remaining);
                std::tuple<Types...> values;
                bool success = true;
                
                std::apply([&](auto&... args) {
                    ((success = success && parse_single_value(ss, args)), ...);
                }, values);

                if (!success) {
                    result.clear();
                    return result;
                }

                result.push_back(values);
                
                // Get remaining data
                std::string rest;
                std::getline(ss, rest);
                remaining = trim(rest);
            }
            
            return result;
        }
    }

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
        //contains elements of equal values 
        std::stable_sort(idx.begin(), idx.end(),
            [&data](size_t i1, size_t i2) {return data[i1] < data[i2];});
        return idx;
    }
}}