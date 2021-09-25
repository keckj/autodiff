#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u, ux, ua, ub, uc;
};

// A type defining parameters for a function of interest
// As var cannot directly be passed by argument to cuda kernels
// we store type T and cast it later to Variable<T>.
// Note that 'var' corresponds to Variable<double>.
template <typename P>
struct Params
{
    P a, b, c;
    __host__ __device__ auto as_variables() const { return Params<Variable<P>>{a,b,c}; }
};

// The function that depends on parameters for which derivatives are needed
__host__ __device__ var f(var x, const Params<var>& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

__global__ void kernel(double xval, Params<double> pval, T* output) {
    var x(xval);                          // the input x variable
    Params<var> p = pval.as_variables();  // the input parameter variables
    var u = f(x, p);                      // the output variable u
    auto [ux, ua, ub, uc] = derivatives(u, wrt(x, p.a, p.b, p.c)); // evaluate the derivatives of u with respect to x and parameters a, b, c
    output[0] = T{double(u), ux, ua, ub, uc};
}
AUTODIFF_NVRTC_END_SOURCE


static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.ux, y.ux) && ::allclose(x.ua, y.ua) && ::allclose(x.ub, y.ub) && ::allclose(x.uc, y.uc);
}


int main()
{
    double xval = 0.5;   // the input x value, will be converted to var later

    Params<double> pval; // initialize the parameter variables values
    pval.a = 1.0;        // the parameter a of type double, will be converted to var later
    pval.b = 2.0;        // the parameter b of type double, will be converted to var later
    pval.c = 3.0;        // the parameter c of type double, will be converted to var later

    T h_results; 
    {
        var x(xval);                          // the input x variable
        Params<var> p = pval.as_variables();  // the input parameter variables
        var u = f(x, p);                      // the output variable u
        auto [ux, ua, ub, uc] = derivatives(u, wrt(x, p.a, p.b, p.c)); // evaluate the derivatives of u with respect to x and parameters a, b, c
        h_results = T{double(u), ux, ua, ub, uc};
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(xval, pval, output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&xval, &pval, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    print_example_header(__FILE__);

    std::cout << "HOST" << std::endl;
    std::cout << "u  = " << h_results.u  << std::endl;  // print the evaluated output u
    std::cout << "ux = " << h_results.ux << std::endl;  // print the evaluated derivative du/dx
    std::cout << "ua = " << h_results.ua << std::endl;  // print the evaluated derivative du/da
    std::cout << "ub = " << h_results.ub << std::endl;  // print the evaluated derivative du/db
    std::cout << "uc = " << h_results.uc << std::endl;  // print the evaluated derivative du/dc
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u  = " << d_results.u  << std::endl;  // print the evaluated output u
    std::cout << "ux = " << d_results.ux << std::endl;  // print the evaluated derivative du/dx
    std::cout << "ua = " << d_results.ua << std::endl;  // print the evaluated derivative du/da
    std::cout << "ub = " << d_results.ub << std::endl;  // print the evaluated derivative du/db
    std::cout << "uc = " << d_results.uc << std::endl;  // print the evaluated derivative du/dc
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u =  " << n_results.u  << std::endl;  // print the evaluated output u
    std::cout << "ux = " << n_results.ux << std::endl;  // print the evaluated derivative du/dx
    std::cout << "ua = " << n_results.ua << std::endl;  // print the evaluated derivative du/da
    std::cout << "ub = " << n_results.ub << std::endl;  // print the evaluated derivative du/db
    std::cout << "uc = " << n_results.uc << std::endl;  // print the evaluated derivative du/dc
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
