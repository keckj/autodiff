// C++ includes
#include <iostream>
#include <cstdlib>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/forward/real.hpp>
using namespace autodiff;

// The scalar function for which a 4th order directional Taylor series will be computed.
__host__ __device__ real4th f(const real4th& x, const real4th& y, const real4th& z)
{
    return sin(x * y) * cos(x * z) * exp(z);
}

__global__ void kernel(real4th x, real4th y, real4th z, double t, double* output) {
    auto g = taylorseries(f, along(1, 1, 2), at(x, y, z));
    output[0] = g(t);
}

int main()
{
    real4th x = 1.0;                                       // the input vector x
    real4th y = 2.0;                                       // the input vector y
    real4th z = 3.0;                                       // the input vector z

    auto g = taylorseries(f, along(1, 1, 2), at(x, y, z)); // the function g(t) as a 4th order Taylor approximation of f(x + t, y + t, z + 2t)

    double t = 0.1;                                        // the step length used to evaluate g(t), the Taylor approximation of f(x + t, y + t, z + 2t)

    real4th u = f(x + t, y + t, z + 2*t);                  // the exact value of f(x + t, y + t, z + 2t)

    double h_utaylor = g(t);                               // the 4th order Taylor estimate of f(x + t, y + t, z + 2t)
    
    double d_utaylor;
    {
        double* output;
        cudaMalloc(&output, sizeof(double));
        kernel<<<1,1>>>(x,y,z,t,output);
        cudaMemcpy(&d_utaylor, output, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << std::fixed;
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):" << std::endl;
    std::cout << "u(exact)  = " << u << " (expected value: -16.847071)" << std::endl;
    std::cout << "u(taylor) = " << h_utaylor << " on __host__   (expected value: -16.793986)" << std::endl;
    std::cout << "u(taylor) = " << d_utaylor << " on __device__ (expected value: -16.793986)" << std::endl;

    return allclose(h_utaylor, d_utaylor) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*-------------------------------------------------------------------------------------------------
=== Output ===
---------------------------------------------------------------------------------------------------
Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):
u(exact)  = -16.847071
u(taylor) = -16.793986
-------------------------------------------------------------------------------------------------*/
