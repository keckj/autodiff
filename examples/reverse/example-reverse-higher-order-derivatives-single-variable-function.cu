// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <examples/common.hpp>
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

__global__ void kernel() {
    double xv = 0.5;
    var x = xv;               // the input variable x
    var u = sin(x) * cos(x);  // the output variable u

    auto [ux] = derivativesx(u, wrt(x));    // evaluate the first order derivatives of u
    auto [uxx] = derivativesx(ux, wrt(x));  // evaluate the second order derivatives of ux
    
    printf("DEVICE\n");
    printf("u          = %f\n", (double)u);             // print the evaluated output variable u
    printf("ux(autod)  = %f\n", (double)ux);            // print the evaluated first order derivative ux
    printf("ux(exact)  = %f\n", 1 - 2*sin(xv)*sin(xv)); // print the exact first order derivative ux
    printf("uxx(autod) = %f\n", (double)uxx);           // print the evaluated second order derivative uxx
    printf("uxx(exact) = %f\n", -4*cos(xv)*sin(xv));    // print the exact second order derivative uxx
}

int main()
{
    double xv = 0.5;          // the input value
    var x = xv;               // the input variable x
    var u = sin(x) * cos(x);  // the output variable u

    auto [ux] = derivativesx(u, wrt(x));    // evaluate the first order derivatives of u
    auto [uxx] = derivativesx(ux, wrt(x));  // evaluate the second order derivatives of ux

    double ux_exact  = 1 - 2*sin(xv)*sin(xv);
    double uxx_exact = -4*cos(xv)*sin(xv);
    
    cout << "HOST" << endl;
    cout << "u          = " << u << endl;          // print the evaluated output variable u
    cout << "ux(autod)  = " << ux << endl;         // print the evaluated first order derivative ux
    cout << "ux(exact)  = " << ux_exact << endl;   // print the exact first order derivative ux
    cout << "uxx(autod) = " << uxx << endl;        // print the evaluated second order derivative uxx
    cout << "uxx(exact) = " << uxx_exact << endl;  // print the exact second order derivative uxx
    cout << endl;

    kernel<<<1,1>>>();
    cudaStreamSynchronize(0);

    return allclose((double)ux, ux_exact) && allclose((double)uxx, uxx_exact) ? EXIT_SUCCESS : EXIT_FAILURE;
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
