// C++ includes
#include <iostream>
using namespace std;

// autodiff include
#include <examples/common.hpp>
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// A simple struct to hold and compare our results
struct T { 
    double x0, u0, ux0, uy0;
    double x1, u1, ux1, uy1;
    static bool allclose(const T& x, const T& y) { 
        return (::allclose(x.x0, y.x0) && ::allclose(x.u0, y.u0) && ::allclose(x.ux0, y.ux0) && ::allclose(x.uy0, y.uy0) && 
                ::allclose(x.x1, y.x1) && ::allclose(x.u1, y.u1) && ::allclose(x.ux1, y.ux1) && ::allclose(x.uy1, y.uy1));
    }
};


// A two-variable piecewise function for which derivatives are needed
__host__ __device__ var f(var x, var y) { 
    return condition(x < y, x * y, x * x);
}
    
__host__ __device__ T compute_result() {
    T res;
    {
        var x = 1.0;      // the input variable x
        var y = 2.0;      // the input variable y
        var u = f(x, y);  // the output variable u
        auto [ux, uy] = derivatives(u, wrt(x, y)); // evaluate the derivatives of u

        res.x0  = (double)x;
        res.u0  = (double)u;
        res.ux0 = ux;
        res.uy0 = uy;
        
        x.update(3.0); // Change the value of x in the expression tree
        u.update();    // Update the expression tree in a sweep
        cuda::std::tie(ux, uy) = derivatives(u, wrt(x, y)); // re-evaluate the derivatives
        
        res.x1  = (double)x;
        res.u1  = (double)u;
        res.ux1 = ux;
        res.uy1 = uy;
    }
    return res;
}

__global__ void kernel(T* output) {
    output[0] = compute_result();
}

int main()
{
    T h_results = compute_result();
    
    T d_results;
    {
        T* output;
        cudaMalloc(&output, sizeof(T));
        kernel<<<1,1>>>(output);
        cudaMemcpy(&d_results, output, sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(output);
    }

    cout << "HOST" << std::endl;
    cout << "x  = " << h_results.x0  << " (expected 1)" << endl;
    cout << "u  = " << h_results.u0  << " (expected 2)" << endl;  // x = 1, y = 2, so x < y, so x * y = 2
    cout << "ux = " << h_results.ux0 << " (expected 2)" << endl;  // d/dx x * y = y = 2
    cout << "uy = " << h_results.uy0 << " (expected 1)" << endl;  // d/dy x * y = x = 1
    cout << endl;
    cout << "x  = " << h_results.x1  << " (expected 3)" << endl;
    cout << "u  = " << h_results.u1  << " (expected 9)" << endl;  // Now x > y, so x * x = 9
    cout << "ux = " << h_results.ux1 << " (expected 6)" << endl;  // d/dx x * x = 2x = 6
    cout << "uy = " << h_results.uy1 << " (expected 0)" << endl;  // d/dy x * x = 0

    std::cout << std::endl;
    std::cout << "DEVICE" << std::endl;
    cout << "x  = " << d_results.x0  << " (expected 1)" << endl;
    cout << "u  = " << d_results.u0  << " (expected 2)" << endl;  // x = 1, y = 2, so x < y, so x * y = 2
    cout << "ux = " << d_results.ux0 << " (expected 2)" << endl;  // d/dx x * y = y = 2
    cout << "uy = " << d_results.uy0 << " (expected 1)" << endl;  // d/dy x * y = x = 1
    cout << endl;
    cout << "x  = " << d_results.x1  << " (expected 3)" << endl;
    cout << "u  = " << d_results.u1  << " (expected 9)" << endl;  // Now x > y, so x * x = 9
    cout << "ux = " << d_results.ux1 << " (expected 6)" << endl;  // d/dx x * x = 2x = 6
    cout << "uy = " << d_results.uy1 << " (expected 0)" << endl;  // d/dy x * x = 0
    
    return T::allclose(h_results, d_results) ? EXIT_SUCCESS : EXIT_FAILURE;
}
