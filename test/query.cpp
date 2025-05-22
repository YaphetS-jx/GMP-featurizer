#include <gtest/gtest.h>
#include "query.hpp"
#include "resources.hpp"
#include "math.hpp"
#include "atom.hpp"
#include "geometry.hpp"
#include <memory>
#include <cmath>

using namespace gmp::query;
using namespace gmp::atom;
using namespace gmp::math;
using namespace gmp::resources;
using namespace gmp::geometry;
using namespace gmp::containers;



int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 