// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// The single-variable function for which derivatives are needed
__host__ __device__ var f(var x)
{
    return 1.0 + x + x*x + 1.0/x + log(x);
}

int main()
{
    var x = 2.0;   // the input variable x
    var u = f(x);  // the output variable u

    auto [ux] = derivatives(u, wrt(x)); // evaluate the derivative of u with respect to x

    std::cout << "u = " << u << std::endl;  // print the evaluated output variable u
    std::cout << "ux = " << ux << std::endl;  // print the evaluated derivative ux
}
