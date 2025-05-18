#pragma once

namespace gmp { 
    namespace resources {
        template <typename T> struct pool_allocator;
        template <typename T, typename Pool> class pool_unique_ptr;
    }

    namespace atom {
        class atom_t;
        class atom_system_t;
    }

    namespace geometry {
        class lattice_t;
        template <typename T> struct point3d_t;
    }

    namespace math {
        template <typename T> class array3d_t;
        template <typename T> class matrix3d_t;
        template <typename T> class sym_matrix3d_t;
    }

    namespace input {
        class input_t;
        struct descriptor_config_t;
        struct reference_config_t;
    }

    namespace featurizer {
        class featurizer_t;
    }
}