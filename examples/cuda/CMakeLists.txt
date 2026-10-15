# CUDA examples: each example computes its results on the host, on the device with nvcc and on the device with NVRTC
# (the code between AUTODIFF_NVRTC_BEGIN_SOURCE and AUTODIFF_NVRTC_END_SOURCE is compiled at runtime) and checks that
# they match (see common.hpp).
find_package(CUDAToolkit REQUIRED)

# Eigen headers patched for NVRTC (sets AUTODIFF_NVRTC_EIGEN_DIR)
include(AutodiffNvrtcEigen)

# The CUDA include directory given to NVRTC (for the libcu++ headers <cuda/std/...>)
list(GET CUDAToolkit_INCLUDE_DIRS 0 AUTODIFF_EXAMPLES_CUDA_INCLUDE_DIR)

file(GLOB_RECURSE CUFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)
list(SORT CUFILES)

set(CUDA_EXAMPLES)

foreach(CUFILE ${CUFILES})
    get_filename_component(CUNAME ${CUFILE} NAME_WE)
    set(TARGET cuda-${CUNAME}) # the upstream examples/forward and examples/reverse targets use the plain names
    add_executable(${TARGET} ${CUFILE})
    target_link_libraries(${TARGET} PRIVATE autodiff::autodiff Eigen3::Eigen CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(${TARGET} PRIVATE
        AUTODIFF_EXAMPLES_AUTODIFF_INCLUDE_DIR="${PROJECT_SOURCE_DIR}"
        AUTODIFF_EXAMPLES_EIGEN_INCLUDE_DIR="${AUTODIFF_NVRTC_EIGEN_DIR}"
        AUTODIFF_EXAMPLES_CUDA_INCLUDE_DIR="${AUTODIFF_EXAMPLES_CUDA_INCLUDE_DIR}")
    # Calling a host function from device code is only a warning but aborts the kernel at runtime
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=20011,20014>)
    set_target_properties(${TARGET} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
    list(APPEND CUDA_EXAMPLES ${TARGET})
endforeach()

# Run all the CUDA examples (stops at the first failing one)
set(RUN_CUDA_EXAMPLES)
foreach(TARGET ${CUDA_EXAMPLES})
    list(APPEND RUN_CUDA_EXAMPLES COMMAND $<TARGET_FILE:${TARGET}>)
endforeach()
add_custom_target(cuda-examples
    ${RUN_CUDA_EXAMPLES}
    DEPENDS ${CUDA_EXAMPLES}
    COMMENT "Running the CUDA examples..."
    VERBATIM
    USES_TERMINAL)
