#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

__global__ void kernel() {
    double xv = 0.5;
    var x = xv;                             // the input variable x
    var u = sin(x) * cos(x);                // the output variable u

    auto [ux] = derivativesx(u, wrt(x));    // evaluate the first order derivatives of u
    auto [uxx] = derivativesx(ux, wrt(x));  // evaluate the second order derivatives of ux

#ifdef __CUDACC_RTC__
    printf("NVRTC\n");
#else
    printf("DEVICE\n");
#endif
    printf("u          = %f\n", (double)u);             // print the evaluated output variable u
    printf("ux(autod)  = %f\n", (double)ux);            // print the evaluated first order derivative ux
    printf("ux(exact)  = %f\n", 1 - 2*sin(xv)*sin(xv)); // print the exact first order derivative ux
    printf("uxx(autod) = %f\n", (double)uxx);           // print the evaluated second order derivative uxx
    printf("uxx(exact) = %f\n", -4*cos(xv)*sin(xv));    // print the exact second order derivative uxx
    printf("\n");
}
AUTODIFF_NVRTC_END_SOURCE


int main()
{
    double xv = 0.5;                        // the input value
    var x = xv;                             // the input variable x
    var u = sin(x) * cos(x);                // the output variable u

    auto [ux] = derivativesx(u, wrt(x));    // evaluate the first order derivatives of u
    auto [uxx] = derivativesx(ux, wrt(x));  // evaluate the second order derivatives of ux

    double ux_exact  = 1 - 2*sin(xv)*sin(xv);
    double uxx_exact = -4*cos(xv)*sin(xv);
    
    print_example_header(__FILE__);
    
    std::cout << "HOST" << std::endl;
    std::cout << "u          = " << u << std::endl;          // print the evaluated output variable u
    std::cout << "ux(autod)  = " << ux << std::endl;         // print the evaluated first order derivative ux
    std::cout << "ux(exact)  = " << ux_exact << std::endl;   // print the exact first order derivative ux
    std::cout << "uxx(autod) = " << uxx << std::endl;        // print the evaluated second order derivative uxx
    std::cout << "uxx(exact) = " << uxx_exact << std::endl;  // print the exact second order derivative uxx
    std::cout << std::endl;

    kernel<<<1,1>>>();
    cudaStreamSynchronize(0);
        
    CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
    cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, 0);
    cudaStreamSynchronize(0);

    bool success = allclose((double)ux, ux_exact) && allclose((double)uxx, uxx_exact);

    if (success) {
        std::cout << "SUCCESS, results match with exact valuess." << std::endl;
    }
    else {
        std::cerr << "FAILURE, this is a bug, please open an issue." << std::endl;
    }
    print_banner();
    std::cout << std::endl;

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*===============================================================================
Output:
=================================================================================
u = 0.420735
ux(autodiff) = 0.540302
ux(exact) = 0.540302
uxx(autodiff) = -1.68294
uxx(exact) = -1.68294
===============================================================================*/
