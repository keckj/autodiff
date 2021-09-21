// C++ includes
#include <iostream>

// autodiff include
#define AUTODIFF_ENABLE_IMPLICIT_CONVERSION
#include <examples/common.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T {
    Vector<double,5> F;
    Matrix<double,5,5> Jx;
    Matrix<double,5,3> Jp;
    Matrix<double,5,1> Jq;
    Matrix<double,5,9> JJ;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.F, y.F) && ::allclose(x.Jx, y.Jx) && ::allclose(x.Jq, y.Jq) && ::allclose(x.Jp, y.Jp) && ::allclose(x.JJ, y.JJ);
    }
};


// The vector function with parameters for which the Jacobian is needed
__host__ __device__ Vector<real,5> f(const Vector<real,5>& x, const Vector<real,3>& p, const real& q)
{
    return x * p.sum() * exp(q);
}
    
__global__ void kernel(Vector<real,5> x, Vector<real,3> p, real q, T* output)
{
    Vector<real,5> F;                                         // the output vector F = f(x, p, q) evaluated together with Jacobian below
    output[0].Jx = jacobian(f, wrt(x), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jx = dF/dx
    output[0].Jp = jacobian(f, wrt(p), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jp = dF/dp
    output[0].Jq = jacobian(f, wrt(q), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jq = dF/dq
    output[0].JJ = jacobian(f, wrt(q, p, x), at(x, p, q), F); // evaluate the function and the Jacobian matrix Jqpx = [dF/dq, dF/dp, dF/dx]
    output[0].F = F;
}

int main()
{
    Vector<real,5> x(1, 2, 3, 4, 5);  // x = [1, 2, 3, 4, 5]
    Vector<real,3> p(1, 2, 3);        // p = [1, 2, 3]
    real q = -2;                      // the input parameter q as a single variable

    T h_results;
    {
        Vector<real,5> F;                                         // the output vector F = f(x, p, q) evaluated together with Jacobian below
        h_results.Jx = jacobian(f, wrt(x), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jx = dF/dx
        h_results.Jp = jacobian(f, wrt(p), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jp = dF/dp
        h_results.Jq = jacobian(f, wrt(q), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jq = dF/dq
        h_results.JJ = jacobian(f, wrt(q, p, x), at(x, p, q), F); // evaluate the function and the Jacobian matrix Jqpx = [dF/dq, dF/dp, dF/dx]
        h_results.F = F;
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,p,q,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    std::cout << "F = \n" << h_results.F << std::endl;     // print the evaluated output vector F
    std::cout << "Jx = \n" << h_results.Jx << std::endl;   // print the evaluated Jacobian matrix dF/dx
    std::cout << "Jp = \n" << h_results.Jp << std::endl;   // print the evaluated Jacobian matrix dF/dp
    std::cout << "Jq = \n" << h_results.Jq << std::endl;   // print the evaluated Jacobian matrix dF/dq
    std::cout << "Jqpx = \n" << h_results.JJ << std::endl; // print the evaluated Jacobian matrix [dF/dq, dF/dp, dF/dx]
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "F = \n" << d_results.F << std::endl;     // print the evaluated output vector F
    std::cout << "Jx = \n" << d_results.Jx << std::endl;   // print the evaluated Jacobian matrix dF/dx
    std::cout << "Jp = \n" << d_results.Jp << std::endl;   // print the evaluated Jacobian matrix dF/dp
    std::cout << "Jq = \n" << d_results.Jq << std::endl;   // print the evaluated Jacobian matrix dF/dq
    std::cout << "Jqpx = \n" << d_results.JJ << std::endl; // print the evaluated Jacobian matrix [dF/dq, dF/dp, dF/dx]
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
