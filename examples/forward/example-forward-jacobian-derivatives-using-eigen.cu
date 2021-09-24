#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#define AUTODIFF_ENABLE_IMPLICIT_CONVERSION
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T {
    Vector<double,5> F;
    Matrix<double,5,5> J;
};

// The vector function for which the Jacobian is needed
__host__ __device__ Vector<real,5> f(const Vector<real,5>& x)
{
    return x * x.sum();
}

__global__ void kernel(Vector<real,5> x, T* output)
{
    T d_results;
    {
        Vector<real,5> F;                                     // the output vector F = f(x) evaluated together with Jacobian matrix below
        Matrix<double,5,5> J = jacobian(f, wrt(x), at(x), F); // evaluate the output vector F and the Jacobian matrix dF/dx
        d_results.F = F;
        d_results.J = J;
    }
    output[0] = d_results;
}
AUTODIFF_NVRTC_END_SOURCE
    

static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.F, y.F) && ::allclose(x.J, y.J);
}


int main()
{
    Vector<real,5> x(1, 2, 3, 4, 5);
    
    T h_results;
    {
        Vector<real,5> F;                                     // the output vector F = f(x) evaluated together with Jacobian matrix below
        Matrix<double,5,5> J = jacobian(f, wrt(x), at(x), F); // evaluate the output vector F and the Jacobian matrix dF/dx
        h_results.F = F;
        h_results.J = J;
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x, output);
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
    std::cout << "F = \n" << h_results.F << std::endl;    // print the evaluated output vector F
    std::cout << "J = \n" << h_results.J << std::endl;    // print the evaluated Jacobian matrix dF/dx
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "F = \n" << d_results.F << std::endl;    // print the evaluated output vector F
    std::cout << "J = \n" << d_results.J << std::endl;    // print the evaluated Jacobian matrix dF/dx
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "F = \n" << n_results.F << std::endl;    // print the evaluated output vector F
    std::cout << "J = \n" << n_results.J << std::endl;    // print the evaluated Jacobian matrix dF/dx
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
