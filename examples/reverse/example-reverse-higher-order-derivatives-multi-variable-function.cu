#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T {
    double u;
    cuda::std::array<double, 3> du;
    cuda::std::array<cuda::std::array<double, 3>, 3> d2u;
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
AUTODIFF_NVRTC_END_SOURCE


static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.du, y.du) && ::allclose(x.d2u, y.d2u);
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
    
    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        double xval(x), yval(y), zval(z);
        void *args[] = {&xval, &yval, &zval, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    print_example_header(__FILE__);

    std::cout << "HOST" << std::endl;
    std::cout << "u = "   << h_results.u << std::endl;          // print the evaluated output variable u
    std::cout << "ux = "  << h_results.du[0] << std::endl;      // print the evaluated first order derivative ux
    std::cout << "uy = "  << h_results.du[1] << std::endl;      // print the evaluated first order derivative uy
    std::cout << "uz = "  << h_results.du[2] << std::endl;      // print the evaluated first order derivative uz
    std::cout << "uxx = " << h_results.d2u[0][0] << std::endl;  // print the evaluated second order derivative uxx
    std::cout << "uxy = " << h_results.d2u[0][1] << std::endl;  // print the evaluated second order derivative uxy
    std::cout << "uxz = " << h_results.d2u[0][2] << std::endl;  // print the evaluated second order derivative uxz
    std::cout << "uyx = " << h_results.d2u[1][0] << std::endl;  // print the evaluated second order derivative uyx
    std::cout << "uyy = " << h_results.d2u[1][1] << std::endl;  // print the evaluated second order derivative uyy
    std::cout << "uyz = " << h_results.d2u[1][2] << std::endl;  // print the evaluated second order derivative uyz
    std::cout << "uzx = " << h_results.d2u[2][0] << std::endl;  // print the evaluated second order derivative uzx
    std::cout << "uzy = " << h_results.d2u[2][1] << std::endl;  // print the evaluated second order derivative uzy
    std::cout << "uzz = " << h_results.d2u[2][2] << std::endl;  // print the evaluated second order derivative uzz
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = "   << d_results.u << std::endl;          // print the evaluated output variable u
    std::cout << "ux = "  << d_results.du[0] << std::endl;      // print the evaluated first order derivative ux
    std::cout << "uy = "  << d_results.du[1] << std::endl;      // print the evaluated first order derivative uy
    std::cout << "uz = "  << d_results.du[2] << std::endl;      // print the evaluated first order derivative uz
    std::cout << "uxx = " << d_results.d2u[0][0] << std::endl;  // print the evaluated second order derivative uxx
    std::cout << "uxy = " << d_results.d2u[0][1] << std::endl;  // print the evaluated second order derivative uxy
    std::cout << "uxz = " << d_results.d2u[0][2] << std::endl;  // print the evaluated second order derivative uxz
    std::cout << "uyx = " << d_results.d2u[1][0] << std::endl;  // print the evaluated second order derivative uyx
    std::cout << "uyy = " << d_results.d2u[1][1] << std::endl;  // print the evaluated second order derivative uyy
    std::cout << "uyz = " << d_results.d2u[1][2] << std::endl;  // print the evaluated second order derivative uyz
    std::cout << "uzx = " << d_results.d2u[2][0] << std::endl;  // print the evaluated second order derivative uzx
    std::cout << "uzy = " << d_results.d2u[2][1] << std::endl;  // print the evaluated second order derivative uzy
    std::cout << "uzz = " << d_results.d2u[2][2] << std::endl;  // print the evaluated second order derivative uzz
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = "   << n_results.u << std::endl;          // print the evaluated output variable u
    std::cout << "ux = "  << n_results.du[0] << std::endl;      // print the evaluated first order derivative ux
    std::cout << "uy = "  << n_results.du[1] << std::endl;      // print the evaluated first order derivative uy
    std::cout << "uz = "  << n_results.du[2] << std::endl;      // print the evaluated first order derivative uz
    std::cout << "uxx = " << n_results.d2u[0][0] << std::endl;  // print the evaluated second order derivative uxx
    std::cout << "uxy = " << n_results.d2u[0][1] << std::endl;  // print the evaluated second order derivative uxy
    std::cout << "uxz = " << n_results.d2u[0][2] << std::endl;  // print the evaluated second order derivative uxz
    std::cout << "uyx = " << n_results.d2u[1][0] << std::endl;  // print the evaluated second order derivative uyx
    std::cout << "uyy = " << n_results.d2u[1][1] << std::endl;  // print the evaluated second order derivative uyy
    std::cout << "uyz = " << n_results.d2u[1][2] << std::endl;  // print the evaluated second order derivative uyz
    std::cout << "uzx = " << n_results.d2u[2][0] << std::endl;  // print the evaluated second order derivative uzx
    std::cout << "uzy = " << n_results.d2u[2][1] << std::endl;  // print the evaluated second order derivative uzy
    std::cout << "uzz = " << n_results.d2u[2][2] << std::endl;  // print the evaluated second order derivative uzz
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
