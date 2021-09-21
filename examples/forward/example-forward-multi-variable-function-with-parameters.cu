// C++ includes
#include <iostream>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T { 
    double u, dudx, duda, dudb, dudc;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.dudx, y.dudx) && ::allclose(x.duda, y.duda) && ::allclose(x.dudb, y.dudb) && ::allclose(x.dudc, y.dudc);
    }
};

// A type defining parameters for a function of interest
struct Params
{
    dual a;
    dual b;
    dual c;
};

// The function that depends on parameters for which derivatives are needed
__host__ __device__ dual f(dual x, const Params& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

__global__ void kernel(dual x, Params params, T* output)
{
    dual u = f(x, params);                                        // the output variable u
    output[0].dudx = derivative(f, wrt(x), at(x, params));        // evaluate the derivative du/dx
    output[0].duda = derivative(f, wrt(params.a), at(x, params)); // evaluate the derivative du/da
    output[0].dudb = derivative(f, wrt(params.b), at(x, params)); // evaluate the derivative du/db
    output[0].dudc = derivative(f, wrt(params.c), at(x, params)); // evaluate the derivative du/dc
    output[0].u = u.val;
}

int main()
{
    Params params;   // initialize the parameter variables
    params.a = 1.0;  // the parameter a of type dual, not double!
    params.b = 2.0;  // the parameter b of type dual, not double!
    params.c = 3.0;  // the parameter c of type dual, not double!

    dual x = 0.5;    // the input variable x

    T h_results;
    {
        dual u = f(x, params);                                        // the output variable u
        h_results.dudx = derivative(f, wrt(x), at(x, params));        // evaluate the derivative du/dx
        h_results.duda = derivative(f, wrt(params.a), at(x, params)); // evaluate the derivative du/da
        h_results.dudb = derivative(f, wrt(params.b), at(x, params)); // evaluate the derivative du/db
        h_results.dudc = derivative(f, wrt(params.c), at(x, params)); // evaluate the derivative du/dc
        h_results.u = u.val;
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,params,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << " (expected 3.4968)" << std::endl;           // print the evaluated output u
    std::cout << "du/dx = " << h_results.dudx << " (expected 1.53964)" << std::endl;   // print the evaluated derivative du/dx
    std::cout << "du/da = " << h_results.duda << " (expected 0.479426)" << std::endl;  // print the evaluated derivative du/da
    std::cout << "du/db = " << h_results.dudb << " (expected 0.877583)" << std::endl;  // print the evaluated derivative du/db
    std::cout << "du/dc = " << h_results.dudc << " (expected 0.420735)" << std::endl;  // print the evaluated derivative du/dc
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << " (expected 3.4968)" << std::endl;           // print the evaluated output u
    std::cout << "du/dx = " << d_results.dudx << " (expected 1.53964)" << std::endl;   // print the evaluated derivative du/dx
    std::cout << "du/da = " << d_results.duda << " (expected 0.479426)" << std::endl;  // print the evaluated derivative du/da
    std::cout << "du/db = " << d_results.dudb << " (expected 0.877583)" << std::endl;  // print the evaluated derivative du/db
    std::cout << "du/dc = " << d_results.dudc << " (expected 0.420735)" << std::endl;  // print the evaluated derivative du/dc
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
This example would also work if real was used instead of dual. Should you
need higher-order cross derivatives, however, e.g.,:

    double d2udxda = derivative(f, wrt(x, params.a), at(x, params));

then higher-order dual types are the right choicesince real types are
optimally designed for higher-order directional derivatives.
-------------------------------------------------------------------------------------------------*/
