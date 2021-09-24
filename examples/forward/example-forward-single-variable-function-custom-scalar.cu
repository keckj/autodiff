#include <iostream>
#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <cuda/std/complex>
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// Specialize isArithmetic for complex to make it compatible with dual
namespace autodiff::detail {
    template<typename T> struct ArithmeticTraits<cuda::std::complex<T>> : ArithmeticTraits<T> {};
} /* end of namespace autodiff::detail */

using cxdual = Dual<cuda::std::complex<double>, cuda::std::complex<double>>;

// A simple struct to hold our results
struct T { 
    cuda::std::complex<double> u, dudx;
};

// The single-variable function for which derivatives are needed
__host__ __device__ cxdual f(cxdual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

__global__ void kernel(cxdual x, T* output)
{
    cxdual u = f(x);                                // the output variable u
    output[0].u    = u.val;
    output[0].dudx = derivative(f, wrt(x), at(x));  // evaluate the derivative du/dx
}
AUTODIFF_NVRTC_END_SOURCE
    

// Utility functions to handle complex numbers
template<typename T>
std::ostream& operator<<(std::ostream& os, const cuda::std::complex<T>& v)
{
    os << v.real() << " + " << v.imag() << "i";
    return os;
}

static bool allclose(const cuda::std::complex<double>& x, const cuda::std::complex<double>& y, size_t n=1)
{
    return allclose(x.real(), y.real(), n) && allclose(x.imag(), y.imag(), n);
}

static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.dudx, y.dudx);
}


int main()
{
    cxdual x = cuda::std::complex<double>(2.0,1.0);     // the input variable x

    T h_results;
    {
        cxdual u = f(x);                                // the output variable u
        h_results.u = u.val;                            // the value of the output variable
        h_results.dudx = derivative(f, wrt(x), at(x));  // evaluate the derivative du/dx
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&x, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    print_example_header(__FILE__);

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << " (expected 7.20472 + 5.26365i)" << std::endl;   // print the evaluated output u
    std::cout << "du/dx = " << h_results.dudx << " (expected 5.28 + 1.96i)" << std::endl;  // print the evaluated derivative du/dx
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << " (expected 7.20472 + 5.26365i)" << std::endl;   // print the evaluated output u
    std::cout << "du/dx = " << d_results.dudx << " (expected 5.28 + 1.96i)" << std::endl;  // print the evaluated derivative du/dx
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = " << n_results.u << " (expected 7.20472 + 5.26365i)" << std::endl;   // print the evaluated output u
    std::cout << "du/dx = " << n_results.dudx << " (expected 5.28 + 1.96i)" << std::endl;  // print the evaluated derivative du/dx
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
