#pragma once
#include "math.hpp"
#include "types.hpp"

namespace gmp { namespace mcsh {

    using gmp::math::array3d_t;
    using gmp::containers::vector;

    // solid mcsh function registry
    template <typename T>
    class mcsh_registry_t {
    public: 
        using solid_gmp_t = void (*)(const array3d_t<T>&, const T, const T, const T, const T, vector<T>&);

        mcsh_registry_t();
        static mcsh_registry_t& get_instance();
        const solid_gmp_t get_function(int order) const;
        int get_num_values(int order) const;
        
    private: 
        solid_gmp_t functions_[10];
        int num_values_[10];
    };
    
    // functions 
    template <typename T>
    T weighted_square_sum(const int mcsh_order, const vector<T>& v);
}}

#include "mcsh.hxx"