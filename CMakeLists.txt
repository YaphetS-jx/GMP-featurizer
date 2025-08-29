cmake_minimum_required(VERSION 3.14)

# Add CUDA support option early, before project() command
option(ENABLE_CUDA "Enable CUDA support for GPU acceleration" OFF)

# Set project languages conditionally
if(ENABLE_CUDA)
    project(gmp-featurizer LANGUAGES CXX CUDA)
else()
    project(gmp-featurizer LANGUAGES CXX)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add option to select build type
option(BUILD_TYPE_RELEASE "Build in Release mode (default: ON)" ON)
option(BUILD_TYPE_PROFILE "Build in Profile mode for gprof profiling" OFF)

# Add floating-point precision options
option(USE_SINGLE_PRECISION "Use single precision (float) instead of double precision" OFF)

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
    # Find CUDA Toolkit (modern CMake)
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA include and library directories from CUDAToolkit
    set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(CUDA_LIBRARY_DIRS ${CUDAToolkit_LIBRARY_DIR})
    set(CUDA_LIBRARIES CUDA::cudart CUDA::cuda_driver)

    message(STATUS "CUDA enabled - Version: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA library dirs: ${CUDA_LIBRARY_DIRS}")
    message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

    # Set CUDA architecture if specified
    if(DEFINED CUDA_ARCH)
        set(CUDA_ARCH_VALUE "${CUDA_ARCH}")
        message(STATUS "CUDA architecture set to: sm_${CUDA_ARCH}")
    else()
        # Default to 120 if not specified
        set(CUDA_ARCH_VALUE "120")
        message(STATUS "CUDA architecture defaulting to: sm_120")
    endif()

    # Add CUDA definitions
    add_compile_definitions(GMP_ENABLE_CUDA)
    add_compile_definitions(GMP_CUDA_VERSION=${CMAKE_CUDA_COMPILER_VERSION_MAJOR}${CMAKE_CUDA_COMPILER_VERSION_MINOR})
else()
    message(STATUS "CUDA disabled - building CPU-only version")
endif()

# Find and configure RMM if CUDA is enabled
if(ENABLE_CUDA)
    find_package(rmm REQUIRED CONFIG)
    message(STATUS "RMM found: ${rmm_FOUND}")
    message(STATUS "RMM version: ${RMM_VERSION}")
endif()

# Function to set CUDA_ARCHITECTURES for targets
function(set_cuda_architectures_for_target target_name)
    if(ENABLE_CUDA AND DEFINED CUDA_ARCH_VALUE)
        set_target_properties(${target_name} PROPERTIES
            CUDA_ARCHITECTURES ${CUDA_ARCH_VALUE}
        )
    endif()
endfunction()

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
set_cuda_architectures_for_target(gmp-featurizer-lib)

# Add source files to the library
target_sources(gmp-featurizer-lib
    PRIVATE
    src/common/input.cpp
    src/common/atom.cpp
    src/common/geometry.cpp
    src/cpu/featurizer.cpp
    src/cpu/tree.cpp
    src/cpu/region_query.cpp
    src/cpu/cpu_featurizer.cpp
)

# Add CUDA source files when CUDA is enabled
if(ENABLE_CUDA)
    target_sources(gmp-featurizer-lib
        PRIVATE
        src/gpu/cuda_tree.cu
        src/gpu/cuda_region_query.cu
        src/gpu/cuda_featurizer.cu
        src/gpu/gpu_featurizer.cu
    )
endif()

# Set include directories for the library
target_include_directories(gmp-featurizer-lib
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src/common
    ${CMAKE_SOURCE_DIR}/src/cpu
    ${CMAKE_SOURCE_DIR}/src/gpu
    ${CMAKE_SOURCE_DIR}/src/infra
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${GEMMI_INCLUDE_DIR}
)

# Add CUDA include directories if CUDA is enabled
if(ENABLE_CUDA)
    target_include_directories(gmp-featurizer-lib
        PUBLIC
        ${CUDA_INCLUDE_DIRS}
        /usr/local/cuda/include
    )
    
    # Set CUDA properties for the library
    set_target_properties(gmp-featurizer-lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    

    
    # Set CUDA compilation flags for math.cpp to enable GPU qualifiers
    set_source_files_properties(src/common/math.cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

    # Link libraries
target_link_libraries(gmp-featurizer-lib
    PUBLIC
    pthread
)

# Link CUDA and RMM libraries if CUDA is enabled
if(ENABLE_CUDA)
    target_link_libraries(gmp-featurizer-lib
        PUBLIC
        rmm::rmm
        ${CUDA_LIBRARIES}
    )
endif()
    


# Create the main executable
add_executable(gmp-featurizer src/main.cpp)
set_cuda_architectures_for_target(gmp-featurizer)

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

    # Add CUDA test only when CUDA is enabled
    if(ENABLE_CUDA)
        add_executable(gmp-cuda-test
            test/test-cuda.cu
        )
        set_cuda_architectures_for_target(gmp-cuda-test)
        
        add_executable(gmp-rmm-test
            test/rmm_test.cpp
        )
        set_cuda_architectures_for_target(gmp-rmm-test)
        
        add_executable(gmp-cuda-tree-test
            test/cuda_tree_test.cu
        )
        set_cuda_architectures_for_target(gmp-cuda-tree-test)
        
        add_executable(gmp-cuda-region-query-test
            test/cuda_region_query_test.cu
        )
        set_cuda_architectures_for_target(gmp-cuda-region-query-test)
        
    else()
        add_executable(gmp-resources-test
            test/resources.cpp
        )
        set_cuda_architectures_for_target(gmp-resources-test)

        add_executable(gmp-math-test
            test/math.cpp
        )
        set_cuda_architectures_for_target(gmp-math-test)

        add_executable(gmp-geometry-test
            test/geometry.cpp
        )
        set_cuda_architectures_for_target(gmp-geometry-test)

        add_executable(gmp-atom-test
            test/atom.cpp
        )
        set_cuda_architectures_for_target(gmp-atom-test)

        add_executable(gmp-util-test
            test/util.cpp
        )
        set_cuda_architectures_for_target(gmp-util-test)

        add_executable(gmp-input-test
            test/input.cpp
        )
        set_cuda_architectures_for_target(gmp-input-test)

        add_executable(gmp-morton-test
            test/morton_codes.cpp
        )
        set_cuda_architectures_for_target(gmp-morton-test)

        add_executable(gmp-tree-test
            test/tree.cpp
        )
        set_cuda_architectures_for_target(gmp-tree-test)

        add_executable(gmp-region-query-test
            test/region_query.cpp
        )
        set_cuda_architectures_for_target(gmp-region-query-test)

        add_executable(gmp-cif-test
            test/cif_test.cpp
        )
        set_cuda_architectures_for_target(gmp-cif-test)

        add_executable(gmp-cpu-test
            test/gmp_cpu.cpp
        )
        set_cuda_architectures_for_target(gmp-cpu-test)
    endif()

    # Add project root definition to all test executables
    # Add CUDA test definition if CUDA is enabled
    if(ENABLE_CUDA)
        target_compile_definitions(gmp-cuda-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
        target_compile_definitions(gmp-rmm-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
        target_compile_definitions(gmp-cuda-tree-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
        target_compile_definitions(gmp-cuda-region-query-test PRIVATE PROJECT_ROOT="${PROJECT_ROOT}")
    else()
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
    endif()

    # Link against Google Test and the library
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
        
        # Link RMM test
        target_link_libraries(gmp-rmm-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
            rmm::rmm
            ${CUDA_LIBRARIES}
        )

        # Add CUDA include directories and libraries for RMM test
        target_include_directories(gmp-rmm-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CUDA_INCLUDE_DIRS}
        )
        
        # Set CUDA properties for RMM test
        set_target_properties(gmp-rmm-test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        
        # Link CUDA tree test
        target_link_libraries(gmp-cuda-tree-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
            ${CUDA_LIBRARIES}
        )
        
        # Add CUDA include directories and libraries for CUDA tree test
        target_include_directories(gmp-cuda-tree-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CUDA_INCLUDE_DIRS}
        )
        
        # Set CUDA properties for CUDA tree test
        set_target_properties(gmp-cuda-tree-test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        
        # Link CUDA region query test
        target_link_libraries(gmp-cuda-region-query-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
            ${CUDA_LIBRARIES}
        )
        
        # Add CUDA include directories and libraries for CUDA region query test
        target_include_directories(gmp-cuda-region-query-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CUDA_INCLUDE_DIRS}
        )
        
        # Set CUDA properties for CUDA region query test
        set_target_properties(gmp-cuda-region-query-test PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
    
    else()
        target_link_libraries(gmp-resources-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )
        target_include_directories(gmp-resources-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-math-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )
        target_include_directories(gmp-math-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-geometry-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-geometry-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-atom-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-atom-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-util-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-util-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-input-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-input-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-morton-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-morton-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-tree-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-tree-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-region-query-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-region-query-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-cif-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-cif-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

        target_link_libraries(gmp-cpu-test
            PRIVATE
            GTest::gtest_main
            GTest::gmock_main
            gmp-featurizer-lib
        )

        target_include_directories(gmp-cpu-test
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
        )

    endif()

    # Set the output directory for the test executables
    # Set output directory for CUDA test if enabled
    if(ENABLE_CUDA)
        set_target_properties(gmp-cuda-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
        )
        
        set_target_properties(gmp-rmm-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
        )
        
        set_target_properties(gmp-cuda-tree-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
        )
        
        set_target_properties(gmp-cuda-region-query-test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${TEST_RUNTIME_OUTPUT_DIRECTORY}
        )
    else()
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
    endif()

    # Register tests
    include(GoogleTest)

    # Register CUDA test if enabled
    if(ENABLE_CUDA)
        gtest_discover_tests(gmp-cuda-test)
        gtest_discover_tests(gmp-rmm-test)
        gtest_discover_tests(gmp-cuda-tree-test)
        gtest_discover_tests(gmp-cuda-region-query-test)
    else()
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
    endif()

    # Add a custom target to build and run tests based on CUDA availability
    if(ENABLE_CUDA)
        # GPU tests when CUDA is enabled
        add_custom_target(run_all_tests
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            DEPENDS gmp-cuda-test gmp-rmm-test gmp-cuda-tree-test gmp-cuda-region-query-test
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running GPU tests..."
        )
    else()
        # CPU tests when CUDA is disabled
        add_custom_target(run_all_tests
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
            DEPENDS gmp-resources-test gmp-math-test gmp-geometry-test gmp-atom-test gmp-util-test gmp-input-test gmp-morton-test gmp-tree-test gmp-region-query-test gmp-cif-test gmp-cpu-test
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running CPU tests..."
        )
    endif()
endif()
