// C++ includes
#include <iostream>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u, ux;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.ux, y.ux);
    }
};


// The single-variable function for which derivatives are needed
__host__ __device__ var f(var x)
{
    return 1.0 + x + x*x + 1.0/x + log(x);
}

__global__ void kernel(T* output) {
    var x = 2.0;   // the input variable x
    var u = f(x);  // the output variable u
    auto [ux] = derivatives(u, wrt(x)); // evaluate the derivative of u with respect to x
    output[0].u = double(u);
    output[0].ux = double(ux);
}

int main()
{
    T h_results;
    {
        var x = 2.0;   // the input variable x
        var u = f(x);  // the output variable u
        auto [ux] = derivatives(u, wrt(x)); // evaluate the derivative of u with respect to x
        h_results.u = double(u);
        h_results.ux = double(ux);
    }

    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << std::endl;  // print the evaluated output variable u
    std::cout << "ux = " << h_results.ux << std::endl;  // print the evaluated derivative ux
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << std::endl;  // print the evaluated output variable u
    std::cout << "ux = " << d_results.ux << std::endl;  // print the evaluated derivative ux
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
