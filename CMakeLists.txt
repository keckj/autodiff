# The minimum required cmake version
cmake_minimum_required(VERSION 3.0)

# Add cmake modules of this project to the module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Use ccache to speed up repeated compilations
include(CCache)

# Name and details of the project
project(autodiff VERSION 0.6.3 LANGUAGES CXX CUDA)
add_compile_definitions("AUTODIFF_PROJECT_ROOT=${CMAKE_CURRENT_SOURCE_DIR}")

# Enable parallel build if MSVC is used
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

# Include the cmake variables with values for installation directories
include(GNUInstallDirs)

# Ensure proper configuration if in a conda environment
include(CondaAware)

# Define build options
option(AUTODIFF_BUILD_TESTS "Enable the compilation of the test files." ON)
option(AUTODIFF_BUILD_EXAMPLES "Enable the compilation of the example files." ON)
option(AUTODIFF_BUILD_PYTHON "Enable the compilation of the python bindings." OFF)
option(AUTODIFF_BUILD_DOCS "Enable the build of the documentation and website." OFF)

option(USE_SYSTEM_EIGEN "Use system system eigen instead of internal eigen" ON)

# Find CUDA with C++17 support
find_package(CUDAToolkit 11.7 REQUIRED)
add_compile_definitions("AUTODIFF_CUDA_INCLUDE_DIR=${CUDAToolkit_INCLUDE_DIRS}")

# We need to rely on deprecated FindCUDA to determine device architecture
include(FindCUDA)
set(CUDA_ARCH_LIST Auto CACHE STRING
    "List of CUDA architectures (e.g. Pascal, Volta, etc) or \
compute capability versions (6.1, 7.0, etc) to generate code for. \
Set to Auto for automatic detection (default)."
)
cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCH_LIST})
string(REGEX MATCH "[0-9][0-9]" CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_FLAGS_readable}")
add_compile_definitions("AUTODIFF_CUDA_ARCHITECTURE=${CMAKE_CUDA_ARCHITECTURES}")

# Find eigen library with CUDA 11 support
if (USE_SYSTEM_EIGEN)
    message(STATUS "Using system eigen3 library.")
    find_package(Eigen3 3.4 REQUIRED)
else()
    message(STATUS "Using internal eigen3 library.")
    add_compile_definitions(AUTODIFF_USE_INTERNAL_EIGEN)
endif()

add_compile_definitions(EIGEN_MPL2_ONLY)

# autodiff requires a c++17 enabled compiler
set(CMAKE_CXX_STANDARD 17)          # ensure cmake instructs compiler to use C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # ensure the C++ standard given before is actually used
set(CMAKE_CXX_EXTENSIONS OFF)       # avoid compile flags of the type -std=gnu++1z added by cmake

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")

# Build the library (actually, just provide details about the library target, specify required compile options, etc. because autodiff is header-only)
add_subdirectory(autodiff)

# Build the tests
if(AUTODIFF_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Build the python wrappers
if(AUTODIFF_BUILD_PYTHON)
    set(PYBIND11_CPP_STANDARD -std=c++17)  # Ensure pybind11 uses C++17 standard
    find_package(pybind11 REQUIRED)
    add_subdirectory(python)
endif()

# Build the examples
if(AUTODIFF_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Build the docs
if(AUTODIFF_BUILD_DOCS)
    add_subdirectory(docs)
endif()

# Install the cmake config files that permit users to use find_package(autodiff)
include(autodiffInstallCMakeConfigFiles)
