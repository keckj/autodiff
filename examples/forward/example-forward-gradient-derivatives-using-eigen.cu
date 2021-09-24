#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T {
    double u;
    Vector<double,5> g;
};

// The scalar function for which the gradient is needed
__host__ __device__ real f(const Array<real,5>& x)
{
    return (x * x.exp()).sum(); // sum([xi * exp(xi) for i = 1:5])
}

__global__ void kernel(Array<real,5> x, T* output)
{
    T d_results;
    {
        real u;
        d_results.g = gradient(f, wrt(x), at(x), u);
        d_results.u = u.val();
    }
    output[0] = d_results;
}
AUTODIFF_NVRTC_END_SOURCE


static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.g, y.g);
}


int main()
{
    Array<real,5> x(1, 2, 3, 4, 5);

    T h_results;
    {
        real u;
        h_results.g = gradient(f, wrt(x), at(x), u);
        h_results.u = u.val();
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

    std::cout << "u = " << h_results.u << std::endl;                                // print the evaluated output u
    std::cout << "g = " << h_results.g.transpose() << " (on host)" << std::endl;    // print the evaluated gradient vector g = du/dx
    std::cout << "g = " << d_results.g.transpose() << " (on device)" << std::endl;  // print the evaluated gradient vector g = du/dx
    std::cout << "g = " << n_results.g.transpose() << " (nvrtc)" << std::endl;      // print the evaluated gradient vector g = du/dx
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
