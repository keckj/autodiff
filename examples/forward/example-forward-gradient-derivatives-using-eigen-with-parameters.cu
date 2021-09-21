// C++ includes
#include <iostream>

// autodiff include
#include <examples/common.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T {
    double u;
    Vector<double,5> gx;
    Vector<double,3> gp;
    Vector<double,1> gq;
    Vector<double,9> gg;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.gx, y.gx) && ::allclose(x.gp, y.gp) && ::allclose(x.gq, y.gq) && ::allclose(x.gg, y.gg);
    }
};

// The scalar function for which the gradient is needed
__host__ __device__ real f(const Array<real,5>& x, const Array<real,3>& p, const real& q)
{
    return (x * x).sum() * p.sum() * exp(q); // sum([xi * xi for i = 1:5]) * sum(p) * exp(q)
}

__global__ void kernel(Array<real,5> x, Array<real,3> p, real q, T* output) {
    real u;                                                    // the output scalar u = f(x, p, q) evaluated together with gradient below
    output[0].gx = gradient(f, wrt(x), at(x, p, q), u);        // evaluate the function value u and its gradient vector gx = du/dx
    output[0].gp = gradient(f, wrt(p), at(x, p, q), u);        // evaluate the function value u and its gradient vector gp = du/dp
    output[0].gq = gradient(f, wrt(q), at(x, p, q), u);        // evaluate the function value u and its gradient vector gq = du/dq
    output[0].gg = gradient(f, wrt(q, p, x), at(x, p, q), u);  // evaluate the function value u and its gradient vector gqpx = [du/dq, du/dp, du/dx]
    output[0].u = u.val();
}

int main()
{
    using Eigen::VectorXd;

    Array<real,5> x(1, 2, 3, 4, 5); // x = [1, 2, 3, 4, 5]
    Array<real,3> p(1, 2, 3);       // p = [1, 2, 3]
    real q = -2;                    // the input parameter q as a single variable
    
    T h_results;
    {
        real u;                                                   // the output scalar u = f(x, p, q) evaluated together with gradient below
        h_results.gx = gradient(f, wrt(x), at(x, p, q), u);       // evaluate the function value u and its gradient vector gx = du/dx
        h_results.gp = gradient(f, wrt(p), at(x, p, q), u);       // evaluate the function value u and its gradient vector gp = du/dp
        h_results.gq = gradient(f, wrt(q), at(x, p, q), u);       // evaluate the function value u and its gradient vector gq = du/dq
        h_results.gg = gradient(f, wrt(q, p, x), at(x, p, q), u); // evaluate the function value u and its gradient vector gqpx = [du/dq, du/dp, du/dx]
        h_results.u = u.val();
    }
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(x,p,q,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "u = " << h_results.u << " (on host)" << std::endl;                          // print the evaluated output u
    std::cout << "u = " << d_results.u << " (on device)" << std::endl;                        // print the evaluated output u
    std::cout << "du/dx = " << h_results.gx.transpose() << " (on host)" << std::endl;         // print the evaluated gradient vector gx = du/dx
    std::cout << "du/dx = " << d_results.gx.transpose() << " (on device)" << std::endl;       // print the evaluated gradient vector gx = du/dx
    std::cout << "du/dp = " << h_results.gp.transpose() << " (on host)" << std::endl;         // print the evaluated gradient vector gp = du/dp
    std::cout << "du/dp = " << d_results.gp.transpose() << " (on device)" << std::endl;       // print the evaluated gradient vector gp = du/dp
    std::cout << "du/dq = " << h_results.gq.transpose() << " (on host)" << std::endl;         // print the evaluated gradient vector gq = du/dq
    std::cout << "du/dq = " << d_results.gq.transpose() << " (on device)" << std::endl;       // print the evaluated gradient vector gq = du/dq
    std::cout << "du/d(q,p,x) = " << h_results.gg.transpose() << " (on host)" << std::endl;   // print the evaluated gradient vector gqpx = [du/dq, du/dp, du/dx]
    std::cout << "du/d(q,p,x) = " << d_results.gg.transpose() << " (on device)" << std::endl; // print the evaluated gradient vector gqpx = [du/dq, du/dp, du/dx]

    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
This example would also work if dual was used instead. However, if gradient,
Jacobian, and directional derivatives are all you need, then real types are your
best option. You want to use dual types when evaluating higher-order cross
derivatives, which is not supported for real types.
-------------------------------------------------------------------------------------------------*/
