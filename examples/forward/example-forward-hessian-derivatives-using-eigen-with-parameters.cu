#include <examples/common.hpp>

AUTODIFF_NVRTC_BEGIN_SOURCE
#define AUTODIFF_ENABLE_IMPLICIT_CONVERSION
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

// A simple struct to hold our results
struct T {
    double u;
    Vector<double,6> g;
    Matrix<double,6,6> h;
};

// The scalar function for which the Hessian is needed
__host__ __device__ dual2nd f(const Array<dual2nd,3>& x, const Array<dual2nd,2>& p, const dual2nd& q)
{
    return (x * x).sum() + (p * p).sum() * q; // sum(x*x) + sum(p*p) + q
}

__global__ void kernel(Array<dual2nd,3> x, Array<dual2nd,2> p, dual2nd q, T* output) {
    T d_results;
    {
        dual2nd u;         // the output scalar u = f(x) evaluated together with Hessian below
        Vector<dual,6> g;  // gradient of f(x) evaluated together with Hessian below
        Matrix<double,6,6> h = hessian(f, wrt(x, p, q), at(x, p, q), u, g); // evaluate the function value u, its gradient vector g, and its Hessian matrix H with respect to (x, p, q)
        d_results.u = u;
        d_results.g = g;
        d_results.h = h;
    }
    output[0] = d_results;
}
AUTODIFF_NVRTC_END_SOURCE


static bool check_results(const T& x, const T& y) { 
    return ::allclose(x.u, y.u) && ::allclose(x.g, y.g) && ::allclose(x.h, y.h);
}


int main()
{
    Array<dual2nd,3> x(1, 2, 3);
    Array<dual2nd,2> p(4, 5);
    dual2nd q = -2;

    T h_results;
    {
        dual2nd u;         // the output scalar u = f(x) evaluated together with Hessian below
        Vector<dual,6> g;  // gradient of f(x) evaluated together with Hessian below
        Matrix<double,6,6> h = hessian(f, wrt(x, p, q), at(x, p, q), u, g); // evaluate the function value u, its gradient vector g, and its Hessian matrix H with respect to (x, p, q)
        h_results.u = u;
        h_results.g = g;
        h_results.h = h;
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
    std::cout << "u = " << h_results.u << std::endl;   // print the evaluated output u
    std::cout << "g =\n" << h_results.g << std::endl;  // print the evaluated gradient vector g = [du/dx, du/dp, du/dq]
    std::cout << "H = \n" << h_results.h << std::endl; // print the evaluated Hessian matrix H = d²u/d[x, p, q]²
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    std::cout << "u = " << d_results.u << std::endl;   // print the evaluated output u
    std::cout << "g =\n" << d_results.g << std::endl;  // print the evaluated gradient vector g = [du/dx, du/dp, du/dq]
    std::cout << "H = \n" << d_results.h << std::endl; // print the evaluated Hessian matrix H = d²u/d[x, p, q]²
    std::cout << std::endl;
    std::cout << "NVRTC" << std::endl;
    std::cout << "u = " << n_results.u << std::endl;   // print the evaluated output u
    std::cout << "g =\n" << n_results.g << std::endl;  // print the evaluated gradient vector g = [du/dx, du/dp, du/dq]
    std::cout << "H = \n" << n_results.h << std::endl; // print the evaluated Hessian matrix H = d²u/d[x, p, q]²
    
    return exit_example<T>(h_results, d_results, n_results, check_results);
}
