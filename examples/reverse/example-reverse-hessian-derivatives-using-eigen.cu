// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <examples/common.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double u;
    Vector<double, 5> g;
    Matrix<double, 5, 5> H;
    static bool allclose(const T& x, const T& y) { 
        return ::allclose(x.u, y.u) && ::allclose(x.g, y.g) && ::allclose(x.H, y.H);
    }
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
        kernel<<<1,1>>>(xvals,output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    std::cout << "HOST" << std::endl;
    cout << "u = " << h_results.u << endl;    // print the evaluated output variable u
    cout << "g = \n" << h_results.g << endl;  // print the evaluated gradient vector of u
    cout << "H = \n" << h_results.H << endl;  // print the evaluated Hessian matrix of u
    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    cout << "u = " << d_results.u << endl;    // print the evaluated output variable u
    cout << "g = \n" << d_results.g << endl;  // print the evaluated gradient vector of u
    cout << "H = \n" << d_results.H << endl;  // print the evaluated Hessian matrix of u
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
