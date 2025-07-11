#pragma once

namespace gmp {

    // Define floating-point type based on CMake configuration
    #if defined(GMP_USE_SINGLE_PRECISION)
        using gmp_float = float;
        constexpr bool use_single_precision = true;
        constexpr bool use_double_precision = false;
    #else
        using gmp_float = double;
        constexpr bool use_single_precision = false;
        constexpr bool use_double_precision = true;
    #endif

    // Type aliases for convenience
    using flt = gmp_float;
    using flt32 = float;
    using flt64 = double;

} // namespace gmp 