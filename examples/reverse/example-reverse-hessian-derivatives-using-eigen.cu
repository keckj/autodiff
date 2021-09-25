#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u;
    Vector<double, 5> g;
    Matrix<double, 5, 5> H;
};

// The scalar function for which the gradient is needed
__host__ __device__ var f(const Array<var,5>& x)
{
    return sqrt((x * x).sum()); // sqrt(sum([xi * xi for i = 1:5]))
}

__global__ void kernel(Array<double,5> xvals, T* output) {
    T d_results;
    {
        Vector<var,5> x = xvals.cast<var>(); // the input variable x
        var u = f(x);                        // the output variable u
        d_results.H = hessian(u, x, d_results.g);
        d_results.u = (double)u;
    }
    output[0] = d_results;
}
AUTODIFF_NVRTC_END_SOURCE
    

static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.g, y.g) && ::allclose(x.H, y.H);
}


int main()
{
    Vector<double,5> xvals(1, 2, 3, 4, 5);  // x = [1, 2, 3, 4, 5]
    
    T h_results;
    {
        Vector<var,5> x = xvals.cast<var>(); // the input variable x
        var u = f(x);                        // the output variable u
        h_results.H = hessian(u, x, h_results.g);
        h_results.u = (double)u;
    }

    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        cudaDeviceSetLimit(cudaLimitStackSize, 2*1024);
        kernel<<<1,1>>>(xvals, output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
   
    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&xvals, &output};
        cudaDeviceSetLimit(cudaLimitStackSize, 2*1024);
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    print_example_header(__FILE__);

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << std::endl;              // print the evaluated output variable u
    std::cout << "g = " << h_results.g.transpose() << std::endl;  // print the evaluated gradient vector of u
    std::cout << "H = \n" << h_results.H << std::endl;            // print the evaluated Hessian matrix of u
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << std::endl;              // print the evaluated output variable u
    std::cout << "g = " << d_results.g.transpose() << std::endl;  // print the evaluated gradient vector of u
    std::cout << "H = \n" << d_results.H << std::endl;            // print the evaluated Hessian matrix of u
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = " << n_results.u << std::endl;              // print the evaluated output variable u
    std::cout << "g = " << n_results.g.transpose() << std::endl;  // print the evaluated gradient vector of u
    std::cout << "H = \n" << n_results.H << std::endl;            // print the evaluated Hessian matrix of u
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
