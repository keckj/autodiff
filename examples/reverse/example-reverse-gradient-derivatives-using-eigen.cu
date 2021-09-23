// C++ includes
#include <iostream>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u;
    Vector<double,5> du;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.du, y.du);
    }
};

// The scalar function for which the gradient is needed
__host__ __device__ var f(const Array<var,5>& x)
{
    return sqrt((x * x).sum()); // sqrt(sum([xi * xi for i = 1:5]))
}
    
__global__ void kernel(Array<double,5> xvals, T* output)
{
    T d_results;
    {
        Vector<var,5> x = xvals.cast<var>();  // the input variables
        var u = f(x);                         // the output variable y
        d_results.u = (double)u;              // extract output value
        d_results.du = gradient(u, x);        // evaluate the gradient vector dy/dx
    }
    output[0] = d_results;
}

int main()
{
    Vector<double,5> xvals(1, 2, 3, 4, 5);   // x = [1, 2, 3, 4, 5]
    
    T h_results;
    {
        Vector<var,5> x = xvals.cast<var>();   // the input variables
        var u = f(x);                          // the output variable y
        h_results.u = (double)u;               // extract output value
        h_results.du = gradient(u, x);  // evaluate the gradient vector dy/dx
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(xvals, output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << std::endl;         // print the evaluated output y
    std::cout << "du/dx = \n" << h_results.du << std::endl;  // print the evaluated gradient vector du/dx
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << std::endl;         // print the evaluated output y
    std::cout << "du/dx = \n" << d_results.du << std::endl;  // print the evaluated gradient vector du/dx
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
