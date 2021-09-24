#include <iostream>

#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

using T = cuda::std::array<double,5>;

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
__host__ __device__ dual4th f(dual4th x, dual4th y, dual4th z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

__global__ void kernel(dual4th x, dual4th y, dual4th z, T* output) {
    output[0] = derivatives(f, wrt(x, y, x, z), at(x, y, z));
}
AUTODIFF_NVRTC_END_SOURCE


int main()
{
    dual4th x = 1.0;
    dual4th y = 2.0;
    dual4th z = 3.0;
    
    T h_derivatives = derivatives(f, wrt(x, y, x, z), at(x, y, z));
    const auto& [h_u0, h_ux, h_uxy, h_uxyx, h_uxyxz] = h_derivatives;
    
    T d_derivatives;
    const auto& [d_u0, d_ux, d_uxy, d_uxyx, d_uxyxz] = d_derivatives;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x, y, z, output);
        cudaMemcpy(&d_derivatives, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    T n_derivatives;
    const auto& [n_u0, n_ux, n_uxy, n_uxyx, n_uxyxz] = n_derivatives;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&x, &y, &z, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_derivatives, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    print_example_header(__FILE__);
    
    std::cout << "HOST" << std::endl;
    std::cout << "u0 = " << h_u0 << " (expected 27.2113)" << std::endl;         // print the evaluated value of u = f(x, y, z)
    std::cout << "ux = " << h_ux << " (expected 13.6056)" << std::endl;         // print the evaluated derivative du/dx
    std::cout << "uxy = " << h_uxy << " (expected 3.33099)" << std::endl;       // print the evaluated derivative d²u/dxdy
    std::cout << "uxyx = " << h_uxyx << " (expected -0.735916)" << std::endl;   // print the evaluated derivative d³u/dx²dy
    std::cout << "uxyxz = " << h_uxyxz << " (expected 0.074335)" << std::endl;  // print the evaluated derivative d⁴u/dx²dydz
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u0 = " << d_u0 << " (expected 27.2113)" << std::endl;         // print the evaluated value of u = f(x, y, z)
    std::cout << "ux = " << d_ux << " (expected 13.6056)" << std::endl;         // print the evaluated derivative du/dx
    std::cout << "uxy = " << d_uxy << " (expected 3.33099)" << std::endl;       // print the evaluated derivative d²u/dxdy
    std::cout << "uxyx = " << d_uxyx << " (expected -0.735916)" << std::endl;   // print the evaluated derivative d³u/dx²dy
    std::cout << "uxyxz = " << d_uxyxz << " (expected 0.074335)" << std::endl;  // print the evaluated derivative d⁴u/dx²dydz
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u0 = " << n_u0 << " (expected 27.2113)" << std::endl;         // print the evaluated value of u = f(x, y, z)
    std::cout << "ux = " << n_ux << " (expected 13.6056)" << std::endl;         // print the evaluated derivative du/dx
    std::cout << "uxy = " << n_uxy << " (expected 3.33099)" << std::endl;       // print the evaluated derivative d²u/dxdy
    std::cout << "uxyx = " << n_uxyx << " (expected -0.735916)" << std::endl;   // print the evaluated derivative d³u/dx²dy
    std::cout << "uxyxz = " << n_uxyxz << " (expected 0.074335)" << std::endl;  // print the evaluated derivative d⁴u/dx²dydz
    
    return exit_example<T>(h_derivatives, d_derivatives, n_derivatives);
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
In most cases, dual can be replaced by real, as commented in other examples.
However, computing higher-order cross derivatives has definitely to be done
using higher-order dual types (e.g., dual3rd, dual4th)! This is because real
types (e.g., real2nd, real3rd, real4th) are optimally designed for computing
higher-order directional derivatives.
-------------------------------------------------------------------------------------------------*/
