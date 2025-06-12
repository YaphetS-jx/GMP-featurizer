cmake_minimum_required(VERSION 3.14)
project(gmp-featurizer LANGUAGES CXX)

# Set debug mode
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Define project root path
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Replace hardcoded absolute paths with configurable CMake options
# set(BOOST_POOL_INCLUDE_DIR "/media/xx/LEAVE/coding/boost/libs/pool/include")
# set(NLOHMANN_JSON_INCLUDE_DIR "/media/xx/LEAVE/coding/json/include")
set(BOOST_POOL_INCLUDE_DIR "" CACHE PATH "Path to Boost pool include directory")
set(NLOHMANN_JSON_INCLUDE_DIR "" CACHE PATH "Path to nlohmann/json include directory")

# Option to build tests
option(BUILD_TESTS "Build the tests" OFF)

# Create a library target with header files
add_library(gmp-featurizer-lib STATIC)

# Add source files to the library
target_sources(gmp-featurizer-lib
    PRIVATE
    src/atom.cpp
    src/geometry.cpp
    src/error.cpp
    src/input.cpp
    src/featurizer.cpp
    src/query.cpp
    src/math.cpp    
)

# Create the main executable
add_executable(gmp-featurizer src/main.cpp)

# Link the executable against the library
target_link_libraries(gmp-featurizer
    PRIVATE
    gmp-featurizer-lib
)

# Set include directories for the library
target_include_directories(gmp-featurizer-lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${BOOST_POOL_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
)

if(BUILD_TESTS)
    # Enable testing
    enable_testing()
    
    # Include FetchContent for downloading Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # Set output directory for test executables to test folder
    set(TEST_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/test)

    # Create test executables
    add_executable(gmp-resources-test
        test/resources.cpp
    )

    add_executable(gmp-math-test
        test/math.cpp
    )

    add_executable(gmp-geometry-test
        test/geometry.cpp
    )

    add_executable(gmp-atom-test
        test/atom.cpp
    )

    add_executable(gmp-util-test
        test/util.cpp
    )

    add_executable(gmp-input-test
        test/input.cpp
    )

    add_executable(gmp-query-test
        test/query.cpp
    )

    add_executable(gmp-morton-test
        test/morton_codes.cpp
    )

    add_executable(gmp-tree-test
        test/tree.cpp
    )

    add_executable(gmp-region-query-test
        test/region_query.cpp
    )

    # Add project root definition to all test executables
    target_compile_definitions(gmp-resources-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-math-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-geometry-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-atom-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-util-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-input-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-query-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-morton-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-tree-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-region-query-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")

    # Link against Google Test and the library
    target_link_libraries(gmp-resources-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-math-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-geometry-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-atom-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-util-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-input-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-query-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-morton-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-tree-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-region-query-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    # Include directories for tests
    target_include_directories(gmp-resources-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-math-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-geometry-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-atom-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-util-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-input-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-query-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-morton-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-tree-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-region-query-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    # Set the output directory for the test executables
    set_target_properties(gmp-resources-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-math-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-geometry-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-atom-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-util-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-input-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-query-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-morton-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-tree-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-region-query-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    # Register tests
    include(GoogleTest)
    gtest_discover_tests(gmp-resources-test)
    gtest_discover_tests(gmp-math-test)
    gtest_discover_tests(gmp-geometry-test)
    gtest_discover_tests(gmp-atom-test)
    gtest_discover_tests(gmp-util-test)
    gtest_discover_tests(gmp-input-test)
    gtest_discover_tests(gmp-query-test)
    gtest_discover_tests(gmp-morton-test)
    gtest_discover_tests(gmp-tree-test)
    gtest_discover_tests(gmp-region-query-test)

    # Add a custom target to build and run all tests
    add_custom_target(run_all_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS gmp-resources-test gmp-math-test gmp-geometry-test gmp-atom-test gmp-util-test gmp-input-test gmp-query-test gmp-morton-test gmp-tree-test gmp-region-query-test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all tests..."
    )
endif()
