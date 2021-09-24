#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
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
AUTODIFF_NVRTC_END_SOURCE


int main()
{
    real4th x = 1.0;                                       // the input vector x
    real4th y = 2.0;                                       // the input vector y
    real4th z = 3.0;                                       // the input vector z

    auto g = taylorseries(f, along(1, 1, 2), at(x, y, z)); // the function g(t) as a 4th order Taylor approximation of f(x + t, y + t, z + 2t)

    double t = 0.1;                                        // the step length used to evaluate g(t), the Taylor approximation of f(x + t, y + t, z + 2t)

    real4th u = f(x + t, y + t, z + 2*t);                  // the exact value of f(x + t, y + t, z + 2t)

    double h_results = g(t);                               // the 4th order Taylor estimate of f(x + t, y + t, z + 2t)
    
    double d_results;
    {
        double* output;
        cudaMalloc(&output, sizeof(double));
        kernel<<<1,1>>>(x, y, z, t, output);
        cudaMemcpy(&d_results, output, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    double n_results;
    {
        double* output;
        cudaMalloc(&output, sizeof(double));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&x, &y, &z, &t, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    print_example_header(__FILE__);

    std::cout << std::fixed;
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):" << std::endl;
    std::cout << "u(exact)  = " << u << " (expected value: -16.847071)" << std::endl;
    std::cout << "u(taylor) = " << h_results << " on host   (expected value: -16.793986)" << std::endl;
    std::cout << "u(taylor) = " << d_results << " on device (expected value: -16.793986)" << std::endl;
    std::cout << "u(taylor) = " << n_results << " nvrtc     (expected value: -16.793986)" << std::endl;

    return exit_example(h_results, d_results, n_results);
}

/*-------------------------------------------------------------------------------------------------
=== Output ===
---------------------------------------------------------------------------------------------------
Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):
u(exact)  = -16.847071
u(taylor) = -16.793986
-------------------------------------------------------------------------------------------------*/
