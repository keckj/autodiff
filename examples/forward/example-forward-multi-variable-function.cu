// C++ includes
#include <iostream>

// autodiff include
#define AUTODIFF_ENABLE_IMPLICIT_CONVERSION
#include <examples/common.hpp>
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T { 
    double u, dudx, dudy, dudz;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.dudx, y.dudx) && ::allclose(x.dudy, y.dudy) && ::allclose(x.dudz, y.dudz);
    }
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
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
