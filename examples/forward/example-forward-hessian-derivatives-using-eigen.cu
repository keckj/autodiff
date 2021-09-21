// C++ includes
#include <iostream>

// autodiff include
#define AUTODIFF_ENABLE_IMPLICIT_CONVERSION
#include <examples/common.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T {
    double u;
    Vector<double,3> g;
    Matrix<double,3,3> h;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.g, y.g) && ::allclose(x.h, y.h);
    }
};

// The scalar function for which the gradient is needed
__host__ __device__ dual2nd f(const Vector<dual2nd,3>& x)
{
    return x.cwiseProduct(x).sum(); // sum([x(i) * x(i) for i = 1:5])
}

__global__ void kernel(Vector<dual2nd,3> x, T* output) {
    dual2nd u;
    Vector<dual,3> g;
    Matrix<double,3,3> h = hessian(f, wrt(x), at(x), u, g); // evaluate the function value u and its Hessian matrix H
    output[0].u = u;
    output[0].g = g;
    output[0].h = h;
}

int main()
{
    Vector<dual2nd,3> x(1, 2, 3); // x = [1, 2, 3]
    
    T h_results;
    {
        dual2nd u;
        Vector<dual,3> g;
        Matrix<double,3,3> h = hessian(f, wrt(x), at(x), u, g); // evaluate the function value u and its Hessian matrix H
        h_results.u = u;
        h_results.g = g;
        h_results.h = h;
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << std::endl;      // print the evaluated output u
    std::cout << "g = \n" << h_results.g << std::endl;    // print the evaluated gradient vector g = du/dx
    std::cout << "H = \n" << h_results.h << std::endl;    // print the evaluated Hessian matrix H = d²u/dx²
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << std::endl;      // print the evaluated output u
    std::cout << "g = \n" << d_results.g << std::endl;    // print the evaluated gradient vector g = du/dx
    std::cout << "H = \n" << d_results.h << std::endl;    // print the evaluated Hessian matrix H = d²u/dx²
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
