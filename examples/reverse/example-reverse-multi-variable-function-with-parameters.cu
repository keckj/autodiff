// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <examples/common.hpp>
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u, ux, ua, ub, uc;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.ux, y.ux) && ::allclose(x.ua, y.ua) && ::allclose(x.ub, y.ub) && ::allclose(x.uc, y.uc);
    }
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

    std::cout << "HOST" << std::endl;
    cout << "u = "  << h_results.u  << endl;  // print the evaluated output u
    cout << "ux = " << h_results.ux << endl;  // print the evaluated derivative du/dx
    cout << "ua = " << h_results.ua << endl;  // print the evaluated derivative du/da
    cout << "ub = " << h_results.ub << endl;  // print the evaluated derivative du/db
    cout << "uc = " << h_results.uc << endl;  // print the evaluated derivative du/dc
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    cout << "u = "  << d_results.u  << endl;  // print the evaluated output u
    cout << "ux = " << d_results.ux << endl;  // print the evaluated derivative du/dx
    cout << "ua = " << d_results.ua << endl;  // print the evaluated derivative du/da
    cout << "ub = " << d_results.ub << endl;  // print the evaluated derivative du/db
    cout << "uc = " << d_results.uc << endl;  // print the evaluated derivative du/dc
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
