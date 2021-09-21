// C++ includes
#include <iostream>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

__host__ __device__ Array<real4th,5> f(const Array<real4th,5>& x)
{
    return x.sin() / x;
}

__global__ void kernel(Array<real4th,5> x, Array<double,5> v, double t, Array<double,5>* output) {
    auto g = taylorseries(f, along(v), at(x));
    output[0] = g(t);
}

int main()
{
    Array<real4th,5> x(1.0, 2.0, 3.0, 4.0, 5.0);

    Array<double,5> v(1.0, 1.0, 1.0, 1.0, 1.0);

    auto g = taylorseries(f, along(v), at(x)); // the function g(t) as a 4th order Taylor approximation of f(x + t·v)

    double t = 0.1;                            // the step length used to evaluate g(t), the Taylor approximation of f(x + t·v)

    Array<real4th,5> u = f(x + t * v);         // the exact value of f(x + t·v)

    Array<double,5> h_utaylor = g(t);          // the 4th order Taylor estimate of f(x + t·v)
    
    Array<double,5> d_utaylor;
    {
        using T = Array<double,5>;
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,v,t,output);
        cudaMemcpy(&d_utaylor, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << std::fixed;
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t·v):" << std::endl;
    std::cout << "u(exact)  = " << u.transpose() << std::endl;
    std::cout << "u(taylor) = " << h_utaylor.transpose() << " (on host)" << std::endl;
    std::cout << "u(taylor) = " << d_utaylor.transpose() << " (on device)" << std::endl;

    return allclose(h_utaylor, d_utaylor) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*-------------------------------------------------------------------------------------------------
=== Output ===
---------------------------------------------------------------------------------------------------
Comparison between exact evaluation and 4th order Taylor estimate of f(x + t·v):
u(exact)  =  0.810189  0.411052  0.013413 -0.199580 -0.181532
u(taylor) =  0.810189  0.411052  0.013413 -0.199580 -0.181532
-------------------------------------------------------------------------------------------------*/
