#include <iostream>

#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

using T = cuda::std::array<Array<double,5>, 5>;

// The vector function for which higher-order directional derivatives are needed (up to 4th order).
__host__ __device__ Array<real4th,5> f(const Array<real4th,5>& x, real4th p)
{
    return p * x.log();
}

__global__ void kernel(Array<real4th,5> x, real4th p, Array<double,5> vx, double vp, T* output)
{
    output[0] = derivatives(f, along(vx, vp), at(x, p)); // the directional derivatives of f along direction v = (vx, vp) at (x, p)
}
AUTODIFF_NVRTC_END_SOURCE


int main()
{
    Array<real4th,5> x(1.0, 2.0, 3.0, 4.0, 5.0);
    real4th p = 7.0; // the input parameter p = 1

    Array<double,5> vx(1.0, 1.0, 1.0, 1.0, 1.0);
    double vp = 1.0; // the direction vp in the direction vector v = (vx, vp)

    T h_dfdv = derivatives(f, along(vx, vp), at(x, p)); // the directional derivatives of f along direction v = (vx, vp) at (x, p)

    T d_dfdv;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x, p, vx, vp, output);
        cudaMemcpy(&d_dfdv, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    T n_dfdv;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&x, &p, &vx, &vp, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_dfdv, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    print_example_header(__FILE__);

    std::cout << std::scientific << std::showpos;
    std::cout << "Directional derivatives of f along v = (vx, vp) up to 4th order:" << std::endl;
    std::cout << "HOST" << std::endl;
    std::cout << "dfdv[0] = " << h_dfdv[0].transpose() << std::endl; // print the evaluated 0th order directional derivative of f along v (equivalent to f(x, p))
    std::cout << "dfdv[1] = " << h_dfdv[1].transpose() << std::endl; // print the evaluated 1st order directional derivative of f along v
    std::cout << "dfdv[2] = " << h_dfdv[2].transpose() << std::endl; // print the evaluated 2nd order directional derivative of f along v
    std::cout << "dfdv[3] = " << h_dfdv[3].transpose() << std::endl; // print the evaluated 3rd order directional derivative of f along v
    std::cout << "dfdv[4] = " << h_dfdv[4].transpose() << std::endl; // print the evaluated 4th order directional derivative of f along v
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "dfdv[0] = " << d_dfdv[0].transpose() << std::endl; // print the evaluated 0th order directional derivative of f along v (equivalent to f(x, p))
    std::cout << "dfdv[1] = " << d_dfdv[1].transpose() << std::endl; // print the evaluated 1st order directional derivative of f along v
    std::cout << "dfdv[2] = " << d_dfdv[2].transpose() << std::endl; // print the evaluated 2nd order directional derivative of f along v
    std::cout << "dfdv[3] = " << d_dfdv[3].transpose() << std::endl; // print the evaluated 3rd order directional derivative of f along v
    std::cout << "dfdv[4] = " << d_dfdv[4].transpose() << std::endl; // print the evaluated 4th order directional derivative of f along v
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "dfdv[0] = " << n_dfdv[0].transpose() << std::endl; // print the evaluated 0th order directional derivative of f along v (equivalent to f(x, p))
    std::cout << "dfdv[1] = " << n_dfdv[1].transpose() << std::endl; // print the evaluated 1st order directional derivative of f along v
    std::cout << "dfdv[2] = " << n_dfdv[2].transpose() << std::endl; // print the evaluated 2nd order directional derivative of f along v
    std::cout << "dfdv[3] = " << n_dfdv[3].transpose() << std::endl; // print the evaluated 3rd order directional derivative of f along v
    std::cout << "dfdv[4] = " << n_dfdv[4].transpose() << std::endl; // print the evaluated 4th order directional derivative of f along v
    std::cout << std::endl;
    
    double t = 0.1; // the step length along direction v = (vx, vp) used to compute 4th order Taylor estimate of f
    Array<real4th,5> u = f(x + t * vx, p + t * vp); // start from (x, p), walk a step length t = 0.1 along direction v = (vx, vp) and evaluate f at this current point
    Array<double,5> utaylor = h_dfdv[0] + t*h_dfdv[1] + (t*t/2.0)*h_dfdv[2] + (t*t*t/6.0)*h_dfdv[3] + (t*t*t*t/24.0)*h_dfdv[4]; // evaluate the 4th order Taylor estimate of f along direction v = (vx, vp) at a step length of t = 0.1
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate:" << std::endl;
    std::cout << "u(exact)  = " << u.transpose() << std::endl;
    std::cout << "u(taylor) = " << utaylor.transpose() << std::endl;
    
    return exit_example<T>(h_dfdv, d_dfdv, n_dfdv);
}

/*-------------------------------------------------------------------------------------------------
=== Output ===
---------------------------------------------------------------------------------------------------
Directional derivatives of f along v = (vx, vp) up to 4th order:
dfdv[0] = +0.000000e+00 +4.852030e+00 +7.690286e+00 +9.704061e+00 +1.126607e+01
dfdv[1] = +7.000000e+00 +4.193147e+00 +3.431946e+00 +3.136294e+00 +3.009438e+00
dfdv[2] = -5.000000e+00 -7.500000e-01 -1.111111e-01 +6.250000e-02 +1.200000e-01
dfdv[3] = +1.100000e+01 +1.000000e+00 +1.851852e-01 +3.125000e-02 -8.000000e-03
dfdv[4] = -3.400000e+01 -1.625000e+00 -2.222222e-01 -3.906250e-02 -3.200000e-03

Comparison between exact evaluation and 4th order Taylor estimate:
u(exact)  = +6.767023e-01 +5.267755e+00 +8.032955e+00 +1.001801e+01 +1.156761e+01
u(taylor) = +6.766917e-01 +5.267755e+00 +8.032955e+00 +1.001801e+01 +1.156761e+01
-------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
This example would also work if dual was used instead of real. However, real
types are your best option for directional derivatives, as they were optimally
designed for this kind of derivatives.
-------------------------------------------------------------------------------------------------*/
