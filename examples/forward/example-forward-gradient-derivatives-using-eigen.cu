// C++ includes
#include <iostream>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The scalar function for which the gradient is needed
__host__ __device__ real f(const Array<real,5>& x)
{
    return (x * x.exp()).sum(); // sum([xi * exp(xi) for i = 1:5])
}

__global__ void kernel(Array<real,5> x, Vector<double,5>* output)
{
    real u;
    output[0] = gradient(f, wrt(x), at(x), u);
}

int main()
{
    Array<real,5> x(1, 2, 3, 4, 5);

    real u;                                               // the output scalar u = f(x) evaluated together with gradient below

    Vector<double,5> h_g = gradient(f, wrt(x), at(x), u); // evaluate the function value u and its gradient vector g = du/dx
        
    Vector<double,5> d_g;
    {
        using T = decltype(d_g);
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x, output);
        cudaMemcpy(&d_g, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "u = " << u << std::endl;                                  // print the evaluated output u
    std::cout << "g = " << h_g.transpose() << " (on host)" << std::endl;    // print the evaluated gradient vector g = du/dx
    std::cout << "g = " << d_g.transpose() << " (on device)" << std::endl;
    
    return allclose(h_g, d_g) ? EXIT_SUCCESS : EXIT_FAILURE;
}
