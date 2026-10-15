# Provide Eigen headers usable in code compiled at runtime with NVRTC (tests and examples of autodiff).
#
# Eigen needs to be patched to be usable with NVRTC (see ext/eigen-nvrtc/NVRTC.md). By default the patched Eigen 5.0.1
# of the release 5.0.1-nvrtc of https://github.com/keckj/eigen-nvrtc is fetched at configure time; a directory
# containing already patched Eigen headers (for instance an Eigen 5.0.1 checkout to which
# ext/eigen-nvrtc/eigen-5.0.1-nvrtc.patch has been applied) can be given instead with AUTODIFF_NVRTC_EIGEN_INCLUDE_DIR.
# On return, AUTODIFF_NVRTC_EIGEN_DIR is the directory containing Eigen/Core.

set(AUTODIFF_NVRTC_EIGEN_INCLUDE_DIR "" CACHE PATH
    "Directory containing NVRTC compatible Eigen headers (Eigen/Core, ...), fetched if empty")

if(NOT AUTODIFF_NVRTC_EIGEN_INCLUDE_DIR)
    include(FetchContent)
    FetchContent_Declare(eigen_nvrtc
        GIT_REPOSITORY https://github.com/keckj/eigen-nvrtc.git
        GIT_TAG 5.0.1-nvrtc
        GIT_SHALLOW TRUE
        UPDATE_DISCONNECTED TRUE
        SOURCE_SUBDIR do-not-add-subdirectory) # headers only, Eigen's own CMake project is not needed
    FetchContent_MakeAvailable(eigen_nvrtc)
    set(AUTODIFF_NVRTC_EIGEN_DIR "${eigen_nvrtc_SOURCE_DIR}")
else()
    set(AUTODIFF_NVRTC_EIGEN_DIR "${AUTODIFF_NVRTC_EIGEN_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${AUTODIFF_NVRTC_EIGEN_DIR}/Eigen/src/Core/util/NVRTCSupport.h")
    message(FATAL_ERROR "No NVRTC compatible Eigen found in ${AUTODIFF_NVRTC_EIGEN_DIR}")
endif()
message(STATUS "Eigen headers for NVRTC: ${AUTODIFF_NVRTC_EIGEN_DIR}")
