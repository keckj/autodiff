// C++ includes
#include <iostream>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T { 
    double u, dudx;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.dudx, y.dudx);
    }
};

// The single-variable function for which derivatives are needed
__host__ __device__ dual f(dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

__global__ void kernel(dual x, T* output)
{
    dual u = f(x);
    output[0].u = u.val;
    output[0].dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx
}

int main()
{
    dual x = 2.0;                                        // the input variable x
    
    T h_results;
    {
        dual u = f(x);                                   // the output variable u
        h_results.u = u.val;                             // the value of the variable
        h_results.dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << " (expected 8.19315)" << std::endl;     // print the evaluated output u
    std::cout << "du/dx = " << h_results.dudx << " (expected 5.25)" << std::endl; // print the evaluated derivative du/dx
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << " (expected 8.19315)" << std::endl;     // print the evaluated output u
    std::cout << "du/dx = " << d_results.dudx << " (expected 5.25)" << std::endl; // print the evaluated derivative du/dx
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
