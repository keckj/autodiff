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
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.du, y.du);
    }
};

// The multi-variable function for which derivatives are needed
__host__ __device__ var f(var x, var y, var z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

// This kernel exposes UB because var uses virtual inheritance and cannot be used as argument in __global__ functions
__global__ void broken_kernel(var x, var y, var z, T* output) {
//                            ^^^    ^^^    ^^^
    var u = f(x, y, z);                                           // the output variable u
    cuda::std::array<double,3> du = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z
    output[0] = T{double(u), du};
}

// This kernel will work as expected
__global__ void working_kernel(double xval, double yval, double zval, T* output) {
    var x(xval), y(yval), z(zval);
    var u = f(x, y, z);                                           // the output variable u
    cuda::std::array<double,3> du = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z
    output[0] = T{double(u), du};
}

int main()
{
    var x = 1.0;   // the input variable x
    var y = 2.0;   // the input variable y
    var z = 3.0;   // the input variable z
    
    T h_results;
    {
        var u = f(x, y, z);                          // the output variable u
        h_results.u  = double(u);                    // extract output variable value by using explicit cast
        h_results.du = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        working_kernel<<<1,1>>>((double)x,(double)y,(double)z,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    cout << "u = "  << h_results.u     << endl;  // print the evaluated output u
    cout << "ux = " << h_results.du[0] << endl;  // print the evaluated derivative ux
    cout << "uy = " << h_results.du[1] << endl;  // print the evaluated derivative uy
    cout << "uz = " << h_results.du[2] << endl;  // print the evaluated derivative uz
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    cout << "u = "  << d_results.u     << endl;  // print the evaluated output u
    cout << "ux = " << d_results.du[0] << endl;  // print the evaluated derivative ux
    cout << "uy = " << d_results.du[1] << endl;  // print the evaluated derivative uy
    cout << "uz = " << d_results.du[2] << endl;  // print the evaluated derivative uz
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
