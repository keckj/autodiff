find_package(Eigen3 REQUIRED)
find_package(Catch2 3 QUIET)
if(NOT Catch2_FOUND)
    message(STATUS "Catch2 v3 not found, fetching it with FetchContent")
    include(FetchContent)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.8.1
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(Catch2)
endif()

file(GLOB_RECURSE TEST_CPP_FILES "*.test.cpp")
add_executable(autodiff-cpptests main.cpp ${TEST_CPP_FILES})
target_link_libraries(autodiff-cpptests autodiff::autodiff Eigen3::Eigen Catch2::Catch2WithMain)
set_target_properties(autodiff-cpptests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_COMPILER)
    file(GLOB_RECURSE TEST_CUDA_FILES "*.test.cu")
    list(FILTER TEST_CUDA_FILES EXCLUDE REGEX "/nvrtc/") # NVRTC kernels are compiled at runtime by autodiff-nvrtctests
    add_executable(autodiff-cudatests main.cpp ${TEST_CUDA_FILES})
    target_link_libraries(autodiff-cudatests autodiff::autodiff Eigen3::Eigen Catch2::Catch2WithMain)
    set_target_properties(autodiff-cudatests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
    # Calling a host function from device code compiles with a warning but silently aborts the kernel at runtime
    target_compile_options(autodiff-cudatests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=20011,20014>)
    # Spurious warnings of Eigen 3.4 in device code (__host__ __device__ on defaulted functions)
    target_compile_options(autodiff-cudatests PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>")
endif()
# Note: Disabled compilation of implicit conversion tests because they are
# stressing too much CI compilers in addition to the other tests. 
# Allan Leal, 19.06.2023

# add_executable(autodiff-cpptests-implicit-conversion main.cpp ${TEST_CPP_FILES})
# target_compile_definitions(autodiff-cpptests-implicit-conversion PUBLIC AUTODIFF_ENABLE_IMPLICIT_CONVERSION=1)
# target_link_libraries(autodiff-cpptests-implicit-conversion autodiff::autodiff Eigen3::Eigen Catch2::Catch2WithMain)
# set_target_properties(autodiff-cpptests-implicit-conversion PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

if(AUTODIFF_BUILD_NVRTC_TESTS)
    add_subdirectory(nvrtc)
endif()

# Add target tests that performs all C++ and Python tests
if(CMAKE_CUDA_COMPILER)
    add_custom_target(tests
        COMMENT "Running C++ and CUDA tests..."
        COMMAND $<TARGET_FILE:autodiff-cpptests>
        COMMAND $<TARGET_FILE:autodiff-cudatests>
        # COMMAND $<TARGET_FILE:autodiff-cpptests-implicit-conversion>
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
else()
    add_custom_target(tests
            COMMENT "Running C++ tests..."
            COMMAND $<TARGET_FILE:autodiff-cpptests>
            # COMMAND $<TARGET_FILE:autodiff-cpptests-implicit-conversion>
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

if(AUTODIFF_TEST_SANITIZE)
    include(CheckCXXCompilerFlag)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")  # Also needs to be a link flag for test to work
    check_cxx_compiler_flag(-fsanitize=address HAVE_ASAN)
    unset(CMAKE_REQUIRED_FLAGS)
    if(HAVE_ASAN)
        target_compile_options(autodiff-cpptests PRIVATE "-fsanitize=address")
        # target_compile_options(autodiff-cpptests-implicit-conversion PRIVATE "-fsanitize=address")
        target_link_options(autodiff-cpptests PRIVATE "-fsanitize=address")
        # target_link_options(autodiff-cpptests-implicit-conversion PRIVATE "-fsanitize=address")
        # if(CMAKE_CUDA_COMPILER)
        #     target_compile_options(autodiff-cudatests PRIVATE "-fsanitize=address")
        #     target_link_options(autodiff-cudatests PRIVATE "-fsanitize=address")
        # endif()
    endif()
endif()
