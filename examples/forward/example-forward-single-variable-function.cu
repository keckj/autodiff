// C++ includes
#include <iostream>

#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T { 
    double u, dudx;
};

// The single-variable function for which derivatives are needed
__host__ __device__ dual f(dual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

__global__ void kernel(dual x, T* output)
{
    T d_results;
    {
        dual u = f(x);
        d_results.u = u.val;
        d_results.dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx
    }
    output[0] = d_results;
}
AUTODIFF_NVRTC_END_SOURCE


static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.dudx, y.dudx);
}


int main()
{
    dual x = 2.0;                                        // the input variable x
    
    T h_results;
    {
        dual u = f(x);                                   // the output variable u
        h_results.u = u.val;                             // the value of the output variable
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
    
    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&x, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, &args[0], 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    print_example_header(__FILE__);
    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << " (expected 8.19315)" << std::endl;     // print the evaluated output u
    std::cout << "du/dx = " << h_results.dudx << " (expected 5.25)" << std::endl; // print the evaluated derivative du/dx
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << " (expected 8.19315)" << std::endl;     // print the evaluated output u
    std::cout << "du/dx = " << d_results.dudx << " (expected 5.25)" << std::endl; // print the evaluated derivative du/dx
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = " << n_results.u << " (expected 8.19315)" << std::endl;     // print the evaluated output u
    std::cout << "du/dx = " << n_results.dudx << " (expected 5.25)" << std::endl; // print the evaluated derivative du/dx

    return exit_example<T>(h_results, d_results, n_results, check_results);
}
