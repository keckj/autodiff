#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u, ux;
};

// The single-variable function for which derivatives are needed
__host__ __device__ var f(var x)
{
    return 1.0 + x + x*x + 1.0/x + log(x);
}

__global__ void kernel(T* output) {
    var x = 2.0;                        // the input variable x
    var u = f(x);                       // the output variable u
    auto [ux] = derivatives(u, wrt(x)); // evaluate the derivative of u with respect to x
    output[0].u = double(u);            // extract value from variable
    output[0].ux = ux;
}
AUTODIFF_NVRTC_END_SOURCE


static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.ux, y.ux);
}


int main()
{
    T h_results;
    {
        var x = 2.0;                        // the input variable x
        var u = f(x);                       // the output variable u
        auto [ux] = derivatives(u, wrt(x)); // evaluate the derivative of u with respect to x
        h_results.u = double(u);            // extract value from variable
        h_results.ux = ux;
    }

    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    print_example_header(__FILE__);

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << std::endl;    // print the evaluated output variable u
    std::cout << "ux = " << h_results.ux << std::endl;  // print the evaluated derivative ux
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << std::endl;    // print the evaluated output variable u
    std::cout << "ux = " << d_results.ux << std::endl;  // print the evaluated derivative ux
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = " << n_results.u << std::endl;    // print the evaluated output variable u
    std::cout << "ux = " << n_results.ux << std::endl;  // print the evaluated derivative ux
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
