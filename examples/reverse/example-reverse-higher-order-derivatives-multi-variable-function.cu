// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <examples/common.hpp>
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u;
    cuda::std::array<double, 3> du;
    cuda::std::array<cuda::std::array<double, 3>, 3> d2u;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.du, y.du) && ::allclose(x.d2u, y.d2u);
    }
};

__global__ void kernel(double xval, double yval, double zval, T* output) {
    var x(xval);
    var y(yval);
    var z(zval);
        
    var u = x * log(y) * exp(z);  // the output variable u

    T d_results;
    {
        d_results.u = (double)u;

        auto [ux, uy, uz] = derivativesx(u, wrt(x, y, z));  // evaluate the derivatives of u with respect to x, y, z.
        d_results.du = {(double)ux, (double)uy, (double)uz};

        d_results.d2u[0] = derivatives(ux, wrt(x, y, z)); // evaluate the derivatives of ux with respect to x, y, z.
        d_results.d2u[1] = derivatives(uy, wrt(x, y, z)); // evaluate the derivatives of uy with respect to x, y, z.
        d_results.d2u[2] = derivatives(uz, wrt(x, y, z)); // evaluate the derivatives of uz with respect to x, y, z.
    }

    output[0] = d_results;
}

int main()
{
    var x = 1.0;  // the input variable x
    var y = 0.5;  // the input variable y
    var z = 2.0;  // the input variable z
    
    T h_results;
    {
        var u = x * log(y) * exp(z);  // the output variable u
        h_results.u = (double)u;

        auto [ux, uy, uz] = derivativesx(u, wrt(x, y, z));  // evaluate the derivatives of u with respect to x, y, z.
        h_results.du = {(double)ux, (double)uy, (double)uz};

        h_results.d2u[0] = derivatives(ux, wrt(x, y, z)); // evaluate the derivatives of ux with respect to x, y, z.
        h_results.d2u[1] = derivatives(uy, wrt(x, y, z)); // evaluate the derivatives of uy with respect to x, y, z.
        h_results.d2u[2] = derivatives(uz, wrt(x, y, z)); // evaluate the derivatives of uz with respect to x, y, z.
    }

    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>((double)x, (double)y, (double)z, output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    cout << "u = " << h_results.u << endl;  // print the evaluated output variable u

    cout << "ux = " << h_results.du[0] << endl;  // print the evaluated first order derivative ux
    cout << "uy = " << h_results.du[1] << endl;  // print the evaluated first order derivative uy
    cout << "uz = " << h_results.du[2] << endl;  // print the evaluated first order derivative uz

    cout << "uxx = " << h_results.d2u[0][0] << endl;  // print the evaluated second order derivative uxx
    cout << "uxy = " << h_results.d2u[0][1] << endl;  // print the evaluated second order derivative uxy
    cout << "uxz = " << h_results.d2u[0][2] << endl;  // print the evaluated second order derivative uxz

    cout << "uyx = " << h_results.d2u[1][0] << endl;  // print the evaluated second order derivative uyx
    cout << "uyy = " << h_results.d2u[1][1] << endl;  // print the evaluated second order derivative uyy
    cout << "uyz = " << h_results.d2u[1][2] << endl;  // print the evaluated second order derivative uyz

    cout << "uzx = " << h_results.d2u[2][0] << endl;  // print the evaluated second order derivative uzx
    cout << "uzy = " << h_results.d2u[2][1] << endl;  // print the evaluated second order derivative uzy
    cout << "uzz = " << h_results.d2u[2][2] << endl;  // print the evaluated second order derivative uzz
    
    std::cout << std::endl;

    std::cout << "DEVICE" << std::endl;
    cout << "u = " << d_results.u << endl;  // print the evaluated output variable u

    cout << "ux = " << d_results.du[0] << endl;  // print the evaluated first order derivative ux
    cout << "uy = " << d_results.du[1] << endl;  // print the evaluated first order derivative uy
    cout << "uz = " << d_results.du[2] << endl;  // print the evaluated first order derivative uz

    cout << "uxx = " << d_results.d2u[0][0] << endl;  // print the evaluated second order derivative uxx
    cout << "uxy = " << d_results.d2u[0][1] << endl;  // print the evaluated second order derivative uxy
    cout << "uxz = " << d_results.d2u[0][2] << endl;  // print the evaluated second order derivative uxz

    cout << "uyx = " << d_results.d2u[1][0] << endl;  // print the evaluated second order derivative uyx
    cout << "uyy = " << d_results.d2u[1][1] << endl;  // print the evaluated second order derivative uyy
    cout << "uyz = " << d_results.d2u[1][2] << endl;  // print the evaluated second order derivative uyz

    cout << "uzx = " << d_results.d2u[2][0] << endl;  // print the evaluated second order derivative uzx
    cout << "uzy = " << d_results.d2u[2][1] << endl;  // print the evaluated second order derivative uzy
    cout << "uzz = " << d_results.d2u[2][2] << endl;  // print the evaluated second order derivative uzz

    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
