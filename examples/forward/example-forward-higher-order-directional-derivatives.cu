// C++ includes
#include <iostream>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/forward/real.hpp>
using namespace autodiff;

using T = cuda::std::array<double,5>;

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
__host__ __device__ real4th f(real4th x, real4th y, real4th z)
{
    return sin(x) * cos(y) * exp(z);
}

__global__ void kernel(real4th x, real4th y, real4th z, T* output)
{
    output[0] = derivatives(f, along(1.0, 1.0, 2.0), at(x, y, z)); // the directional derivatives of f along direction v = (1, 1, 2) at (x, y, z) = (1, 2, 3)
}

int main()
{
    real4th x = 1.0;
    real4th y = 2.0;
    real4th z = 3.0;

    T h_dfdv = derivatives(f, along(1.0, 1.0, 2.0), at(x, y, z)); // the directional derivatives of f along direction v = (1, 1, 2) at (x, y, z) = (1, 2, 3)

    T d_dfdv;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,y,z,output);
        cudaMemcpy(&d_dfdv, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    std::cout << "dfdv[0] = " << h_dfdv[0] << " (expected -7.03346)" << std::endl; // print the evaluated 0th order directional derivative of f along v (equivalent to f(x, y, z))
    std::cout << "dfdv[1] = " << h_dfdv[1] << " (expected -33.9515)" << std::endl; // print the evaluated 1st order directional derivative of f along v
    std::cout << "dfdv[2] = " << h_dfdv[2] << " (expected -113.341)" << std::endl; // print the evaluated 2nd order directional derivative of f along v
    std::cout << "dfdv[3] = " << h_dfdv[3] << " (expected -249.358)" << std::endl; // print the evaluated 3rd order directional derivative of f along v
    std::cout << "dfdv[4] = " << h_dfdv[4] << " (expected -225.914)" << std::endl; // print the evaluated 4th order directional derivative of f along v
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "dfdv[0] = " << d_dfdv[0] << " (expected -7.03346)" << std::endl; // print the evaluated 0th order directional derivative of f along v (equivalent to f(x, y, z))
    std::cout << "dfdv[1] = " << d_dfdv[1] << " (expected -33.9515)" << std::endl; // print the evaluated 1st order directional derivative of f along v
    std::cout << "dfdv[2] = " << d_dfdv[2] << " (expected -113.341)" << std::endl; // print the evaluated 2nd order directional derivative of f along v
    std::cout << "dfdv[3] = " << d_dfdv[3] << " (expected -249.358)" << std::endl; // print the evaluated 3rd order directional derivative of f along v
    std::cout << "dfdv[4] = " << d_dfdv[4] << " (expected -225.914)" << std::endl; // print the evaluated 4th order directional derivative of f along v

    return allclose(h_dfdv, d_dfdv) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
This example would also work if dual was used instead of real. However, real
types are your best option for directional derivatives, as they were optimally
designed for this kind of derivatives.
-------------------------------------------------------------------------------------------------*/
