cmake_minimum_required(VERSION 3.14)
project(gmp-featurizer LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add option to select build type
option(BUILD_TYPE_RELEASE "Build in Release mode (default: ON)" ON)
option(BUILD_TYPE_PROFILE "Build in Profile mode for gprof profiling" OFF)

# Add floating-point precision options
option(USE_SINGLE_PRECISION "Use single precision (float) instead of double precision" OFF)

# Add CUDA support option
option(ENABLE_CUDA "Enable CUDA support for GPU acceleration" OFF)

# Set floating-point precision based on options
if(USE_SINGLE_PRECISION)
    add_compile_definitions(GMP_USE_SINGLE_PRECISION)
    message(STATUS "Building with single precision (float)")
else()
    add_compile_definitions(GMP_USE_DOUBLE_PRECISION)
    message(STATUS "Building with double precision (double)")
endif()

# CUDA configuration
if(ENABLE_CUDA)
    # Enable CUDA language
    enable_language(CUDA)
    
    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Set CUDA architecture (configurable via command line)
    set(CUDA_ARCH "120" CACHE STRING "CUDA architecture (e.g., 120 for RTX 5080)")
    set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
    
    # Set CUDA variables manually (modern approach)
    set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_ROOT_DIR}/include)
    set(CUDA_LIBRARIES cuda cudart)
    
    message(STATUS "CUDA enabled - Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # Add CUDA definitions
    add_compile_definitions(GMP_ENABLE_CUDA)
    add_compile_definitions(GMP_CUDA_VERSION=${CMAKE_CUDA_COMPILER_VERSION_MAJOR}${CMAKE_CUDA_COMPILER_VERSION_MINOR})
else()
    message(STATUS "CUDA disabled - building CPU-only version")
endif()

# Set release mode and flags based on the option
if(BUILD_TYPE_PROFILE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -pg -march=native" CACHE STRING "Profile flags" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "-pg" CACHE STRING "Profile linker flags" FORCE)
    set(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "-pg" CACHE STRING "Profile shared linker flags" FORCE)
    
    if(ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -pg" CACHE STRING "CUDA Profile flags" FORCE)
    endif()
elseif(BUILD_TYPE_RELEASE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native" CACHE STRING "Release flags" FORCE)
    
    if(ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3" CACHE STRING "CUDA Release flags" FORCE)
    endif()
else()
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG -march=native" CACHE STRING "Debug flags" FORCE)
    
    if(ENABLE_CUDA)
        set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g -G" CACHE STRING "CUDA Debug flags" FORCE)
    endif()
endif()

# Define project root path
set(PROJECT_ROOT ${CMAKE_SOURCE_DIR})

# Set the output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Set paths for header-only libraries
# You can override these with -DGEMMI_INCLUDE_DIR=... and -DNLOHMANN_JSON_INCLUDE_DIR=... when running cmake
set(GEMMI_INCLUDE_DIR "/usr/local/src/gemmi/include" CACHE PATH "Path to GEMMI include directory")
set(NLOHMANN_JSON_INCLUDE_DIR "/usr/local/src/json/single_include" CACHE PATH "Path to nlohmann/json include directory")

# Option to build tests
option(BUILD_TESTS "Build the tests" OFF)

# Create a library target with header files
add_library(gmp-featurizer-lib STATIC)

# Add source files to the library
target_sources(gmp-featurizer-lib
    PRIVATE
    src/input.cpp
    src/atom.cpp
    src/featurizer.cpp
    src/geometry.cpp
    src/tree.cpp
    src/math.cpp
    src/mcsh.cpp
    src/mcsh_kernel.cpp
    src/region_query.cpp
)

# Set include directories for the library
target_include_directories(gmp-featurizer-lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${GEMMI_INCLUDE_DIR}
)

# Add CUDA include directories if CUDA is enabled
if(ENABLE_CUDA)
    target_include_directories(gmp-featurizer-lib
        PUBLIC
        ${CUDA_INCLUDE_DIRS}
    )
    
    # Set CUDA properties for the library
    set_target_properties(gmp-featurizer-lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# Link libraries
target_link_libraries(gmp-featurizer-lib
    PUBLIC
    pthread
)

# Create the main executable
add_executable(gmp-featurizer src/main.cpp)

target_link_libraries(gmp-featurizer
    PRIVATE
    gmp-featurizer-lib
    pthread
)

# Do NOT link CUDA libraries to gmp-featurizer since it's CPU-only

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

    add_executable(gmp-morton-test
        test/morton_codes.cpp
    )

    add_executable(gmp-tree-test
        test/tree.cpp
    )

    add_executable(gmp-region-query-test
        test/region_query.cpp
    )

    add_executable(gmp-cif-test
        test/cif_test.cpp
    )

    add_executable(gmp-cpu-test
        test/gmp_cpu.cpp
    )

    # Add CUDA test only when CUDA is enabled
    if(ENABLE_CUDA)
        add_executable(gmp-cuda-test
            test/test-cuda.cu
        )
    endif()

    # Add project root definition to all test executables
    target_compile_definitions(gmp-resources-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-math-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-geometry-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-atom-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-util-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-input-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")    
    target_compile_definitions(gmp-morton-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-tree-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-region-query-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-cif-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    target_compile_definitions(gmp-cpu-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")

    # Add CUDA test definition if CUDA is enabled
    if(ENABLE_CUDA)
        target_compile_definitions(gmp-cuda-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    endif()

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

    target_link_libraries(gmp-cif-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    target_link_libraries(gmp-cpu-test
        PRIVATE
        GTest::gtest_main
        GTest::gmock_main
        gmp-featurizer-lib
    )

    # Link CUDA test if CUDA is enabled
    if(ENABLE_CUDA)
        target_link_libraries(gmp-cuda-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
            ${CUDA_LIBRARIES}
        )
        
        # Add CUDA include directories and libraries
        target_include_directories(gmp-cuda-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CUDA_INCLUDE_DIRS}
        )
        
        # Set CUDA properties
        set_target_properties(gmp-cuda-test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        
        # Link CUDA libraries
        target_link_libraries(gmp-cuda-test
            PRIVATE
            ${CUDA_LIBRARIES}
        )
    endif()

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

    target_include_directories(gmp-cif-test
        PRIVATE
        ${CMAKE_SOURCE_DIR}/src
    )

    target_include_directories(gmp-cpu-test
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

    set_target_properties(gmp-morton-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-tree-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-region-query-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-cif-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    set_target_properties(gmp-cpu-test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
    )

    # Set output directory for CUDA test if enabled
    if(ENABLE_CUDA)
        set_target_properties(gmp-cuda-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
        )
    endif()

    # Register tests
    include(GoogleTest)
    gtest_discover_tests(gmp-resources-test)
    gtest_discover_tests(gmp-math-test)
    gtest_discover_tests(gmp-geometry-test)
    gtest_discover_tests(gmp-atom-test)
    gtest_discover_tests(gmp-util-test)
    gtest_discover_tests(gmp-input-test)    
    gtest_discover_tests(gmp-morton-test)
    gtest_discover_tests(gmp-tree-test)
    gtest_discover_tests(gmp-region-query-test)
    gtest_discover_tests(gmp-cif-test)
    gtest_discover_tests(gmp-cpu-test)

    # Register CUDA test if enabled
    if(ENABLE_CUDA)
        gtest_discover_tests(gmp-cuda-test)
    endif()

    # Add a custom target to build and run all tests
    add_custom_target(run_all_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS gmp-resources-test gmp-math-test gmp-geometry-test gmp-atom-test gmp-util-test gmp-input-test gmp-morton-test gmp-tree-test gmp-region-query-test gmp-cif-test gmp-cpu-test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running all tests..."
    )

    # Add CUDA test to dependencies if enabled
    if(ENABLE_CUDA)
        add_dependencies(run_all_tests gmp-cuda-test)
    endif()
endif()
