add_subdirectory(forward)
add_subdirectory(reverse)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(cuda)
endif()
