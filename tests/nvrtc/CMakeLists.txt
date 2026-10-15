# Tests of autodiff compiled at runtime with NVRTC (no C++ standard library available in the kernels)
find_package(CUDAToolkit REQUIRED)

# Eigen headers patched for NVRTC (sets AUTODIFF_NVRTC_EIGEN_DIR)
include(AutodiffNvrtcEigen)

add_executable(autodiff-nvrtctests runner.cpp)
target_link_libraries(autodiff-nvrtctests CUDA::nvrtc CUDA::cuda_driver)
set_target_properties(autodiff-nvrtctests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_definitions(autodiff-nvrtctests PRIVATE
    AUTODIFF_NVRTC_AUTODIFF_INCLUDE_DIR="${PROJECT_SOURCE_DIR}"
    AUTODIFF_NVRTC_EIGEN_INCLUDE_DIR="${AUTODIFF_NVRTC_EIGEN_DIR}"
    AUTODIFF_NVRTC_CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}"
    AUTODIFF_NVRTC_TESTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    AUTODIFF_NVRTC_KERNELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
