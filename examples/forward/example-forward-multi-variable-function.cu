#include <iostream>

#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#define AUTODIFF_ENABLE_IMPLICIT_CONVERSION
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T { 
    double u, dudx, dudy, dudz;
};

// The multi-variable function for which derivatives are needed
__host__ __device__ dual f(dual x, dual y, dual z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

__global__ void kernel(dual x, dual y, dual z, T* output) 
{
    output[0].u = f(x, y, z);
    output[0].dudx = derivative(f, wrt(x), at(x, y, z));
    output[0].dudy = derivative(f, wrt(y), at(x, y, z));
    output[0].dudz = derivative(f, wrt(z), at(x, y, z));
}
AUTODIFF_NVRTC_END_SOURCE
    

static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.dudx, y.dudx) && ::allclose(x.dudy, y.dudy) && ::allclose(x.dudz, y.dudz);
}


int main()
{
    dual x = 1.0;
    dual y = 2.0;
    dual z = 3.0;

    T h_results;
    {
        h_results.u = f(x, y, z);
        h_results.dudx = derivative(f, wrt(x), at(x, y, z));
        h_results.dudy = derivative(f, wrt(y), at(x, y, z));
        h_results.dudz = derivative(f, wrt(z), at(x, y, z));
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,y,z,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&x, &y, &z, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    print_example_header(__FILE__);

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << " (expected 27.2113)" << std::endl;         // print the evaluated output u = f(x, y, z)
    std::cout << "du/dx = " << h_results.dudx << " (expected 13.6056)" << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/dy = " << h_results.dudy << " (expected 8.26761)" << std::endl;  // print the evaluated derivative du/dy
    std::cout << "du/dz = " << h_results.dudz << " (expected 5.28638)" << std::endl;  // print the evaluated derivative du/dz
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << " (expected 27.2113)" << std::endl;         // print the evaluated output u = f(x, y, z)
    std::cout << "du/dx = " << d_results.dudx << " (expected 13.6056)" << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/dy = " << d_results.dudy << " (expected 8.26761)" << std::endl;  // print the evaluated derivative du/dy
    std::cout << "du/dz = " << d_results.dudz << " (expected 5.28638)" << std::endl;  // print the evaluated derivative du/dz
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = " << n_results.u << " (expected 27.2113)" << std::endl;         // print the evaluated output u = f(x, y, z)
    std::cout << "du/dx = " << n_results.dudx << " (expected 13.6056)" << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/dy = " << n_results.dudy << " (expected 8.26761)" << std::endl;  // print the evaluated derivative du/dy
    std::cout << "du/dz = " << n_results.dudz << " (expected 5.28638)" << std::endl;  // print the evaluated derivative du/dz
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
