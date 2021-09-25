#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u;
    cuda::std::array<double, 3> du;
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
AUTODIFF_NVRTC_END_SOURCE
    

static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.du, y.du);
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
    
    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__, "working_kernel");
        double xval(x), yval(y), zval(z);
        void *args[] = {&xval, &yval, &zval, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    print_example_header(__FILE__);

    std::cout << "HOST" << std::endl;
    std::cout << "u = "  << h_results.u     << std::endl;  // print the evaluated output u
    std::cout << "ux = " << h_results.du[0] << std::endl;  // print the evaluated derivative ux
    std::cout << "uy = " << h_results.du[1] << std::endl;  // print the evaluated derivative uy
    std::cout << "uz = " << h_results.du[2] << std::endl;  // print the evaluated derivative uz
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = "  << d_results.u     << std::endl;  // print the evaluated output u
    std::cout << "ux = " << d_results.du[0] << std::endl;  // print the evaluated derivative ux
    std::cout << "uy = " << d_results.du[1] << std::endl;  // print the evaluated derivative uy
    std::cout << "uz = " << d_results.du[2] << std::endl;  // print the evaluated derivative uz
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = "  << n_results.u     << std::endl;  // print the evaluated output u
    std::cout << "ux = " << n_results.du[0] << std::endl;  // print the evaluated derivative ux
    std::cout << "uy = " << n_results.du[1] << std::endl;  // print the evaluated derivative uy
    std::cout << "uz = " << n_results.du[2] << std::endl;  // print the evaluated derivative uz
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
