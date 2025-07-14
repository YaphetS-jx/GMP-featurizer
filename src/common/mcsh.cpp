#include "mcsh.hpp"
#include "mcsh_kernel.hpp"

namespace gmp { namespace mcsh {
    // register functions
    template <typename T>
    mcsh_registry_t<T>::mcsh_registry_t() : 
        functions_{ solid_mcsh_0<T>,
                    solid_mcsh_1<T>,
                    solid_mcsh_2<T>,
                    solid_mcsh_3<T>,
                    solid_mcsh_4<T>,
                    solid_mcsh_5<T>,
                    solid_mcsh_6<T>,
                    solid_mcsh_7<T>,
                    solid_mcsh_8<T>,
                    solid_mcsh_9<T>
        },
        num_values_{1, 3, 6, 10, 15, 21, 28, 36, 45} 
    {}

    template <typename T>
    mcsh_registry_t<T>& mcsh_registry_t<T>::get_instance() {
        static mcsh_registry_t<T> instance;
        return instance;
    }

    template <typename T>
    const typename mcsh_registry_t<T>::solid_gmp_t mcsh_registry_t<T>::get_function(int order) const {
        return order < 0 ? functions_[0] : functions_[order];
    }

    template <typename T>
    int mcsh_registry_t<T>::get_num_values(int order) const {
        return order < 0 ? num_values_[0] : num_values_[order];
    }

    // Explicit instantiations for mcsh_registry_t (used externally)
    template class mcsh_registry_t<float>;
    template class mcsh_registry_t<double>;

}} 