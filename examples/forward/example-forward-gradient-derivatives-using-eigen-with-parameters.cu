#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
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
};

// The scalar function for which the gradient is needed
__host__ __device__ real f(const Array<real,5>& x, const Array<real,3>& p, const real& q)
{
    return (x * x).sum() * p.sum() * exp(q); // sum([xi * xi for i = 1:5]) * sum(p) * exp(q)
}

__global__ void kernel(Array<real,5> x, Array<real,3> p, real q, T* output) {
    T d_results;
    {
        real u;                                                    // the output scalar u = f(x, p, q) evaluated together with gradient below
        d_results.gx = gradient(f, wrt(x), at(x, p, q), u);        // evaluate the function value u and its gradient vector gx = du/dx
        d_results.gp = gradient(f, wrt(p), at(x, p, q), u);        // evaluate the function value u and its gradient vector gp = du/dp
        d_results.gq = gradient(f, wrt(q), at(x, p, q), u);        // evaluate the function value u and its gradient vector gq = du/dq
        d_results.gg = gradient(f, wrt(q, p, x), at(x, p, q), u);  // evaluate the function value u and its gradient vector gqpx = [du/dq, du/dp, du/dx]
        d_results.u = u.val();
    }
    output[0] = d_results;
}
AUTODIFF_NVRTC_END_SOURCE


static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.gx, y.gx) && ::allclose(x.gp, y.gp) && ::allclose(x.gq, y.gq) && ::allclose(x.gg, y.gg);
}


int main()
{
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
        kernel<<<1,1>>>(x, p, q, output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }
    
    T n_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        CUfunction nvrtc_kernel = build_nvrtc_kernel(__FILE__);
        void *args[] = {&x, &p, &q, &output};
        cuLaunchKernel(nvrtc_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
        cudaMemcpy(&n_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    print_example_header(__FILE__);

    std::cout << "HOST" << std::endl;
    std::cout << "u = " << h_results.u << std::endl;                          // print the evaluated output u
    std::cout << "du/dx = " << h_results.gx.transpose() << std::endl;         // print the evaluated gradient vector gx = du/dx
    std::cout << "du/dp = " << h_results.gp.transpose() << std::endl;         // print the evaluated gradient vector gp = du/dp
    std::cout << "du/dq = " << h_results.gq.transpose() << std::endl;         // print the evaluated gradient vector gq = du/dq
    std::cout << "du/d(q,p,x) = " << h_results.gg.transpose() << std::endl;   // print the evaluated gradient vector gqpx = [du/dq, du/dp, du/dx]
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << std::endl;                          // print the evaluated output u
    std::cout << "du/dx = " << d_results.gx.transpose() << std::endl;         // print the evaluated gradient vector gx = du/dx
    std::cout << "du/dp = " << d_results.gp.transpose() << std::endl;         // print the evaluated gradient vector gp = du/dp
    std::cout << "du/dq = " << d_results.gq.transpose() << std::endl;         // print the evaluated gradient vector gq = du/dq
    std::cout << "du/d(q,p,x) = " << d_results.gg.transpose() << std::endl;   // print the evaluated gradient vector gqpx = [du/dq, du/dp, du/dx]
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = " << n_results.u << std::endl;                          // print the evaluated output u
    std::cout << "du/dx = " << n_results.gx.transpose() << std::endl;         // print the evaluated gradient vector gx = du/dx
    std::cout << "du/dp = " << n_results.gp.transpose() << std::endl;         // print the evaluated gradient vector gp = du/dp
    std::cout << "du/dq = " << n_results.gq.transpose() << std::endl;         // print the evaluated gradient vector gq = du/dq
    std::cout << "du/d(q,p,x) = " << n_results.gg.transpose() << std::endl;   // print the evaluated gradient vector gqpx = [du/dq, du/dp, du/dx]

    return exit_example<T>(h_results, d_results, n_results, check_results);
}

/*-------------------------------------------------------------------------------------------------
=== Note ===
---------------------------------------------------------------------------------------------------
This example would also work if dual was used instead. However, if gradient,
Jacobian, and directional derivatives are all you need, then real types are your
best option. You want to use dual types when evaluating higher-order cross
derivatives, which is not supported for real types.
-------------------------------------------------------------------------------------------------*/
