// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The vector function with parameters for which the Jacobian is needed
VectorX<real,5> f(const VectorX<real,5>& x, const VectorX<real,3>& p, const real& q)
{
    return x * p.sum() * exp(q);
}

int main()
{
    VectorX<real,5> x(1, 2, 3, 4, 5);  // x = [1, 2, 3, 4, 5]

    VectorX<real,3> p(1, 2, 3);        // p = [1, 2, 3]

    real q = -2;         // the input parameter q as a single variable

    VectorX<real,5> F;   // the output vector F = f(x, p, q) evaluated together with Jacobian below

    MatrixX<double,5,5> Jx  = jacobian(f, wrt(x), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jx = dF/dx
    MatrixX<double,5,3> Jp  = jacobian(f, wrt(p), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jp = dF/dp
    MatrixX<double,5,1> Jq  = jacobian(f, wrt(q), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jq = dF/dq
    MatrixX<double,5,9> Jqpx = jacobian(f, wrt(q, p, x), at(x, p, q), F); // evaluate the function and the Jacobian matrix Jqpx = [dF/dq, dF/dp, dF/dx]

    std::cout << "F = \n" << F << std::endl;     // print the evaluated output vector F
    std::cout << "Jx = \n" << Jx << std::endl;   // print the evaluated Jacobian matrix dF/dx
    std::cout << "Jp = \n" << Jp << std::endl;   // print the evaluated Jacobian matrix dF/dp
    std::cout << "Jq = \n" << Jq << std::endl;   // print the evaluated Jacobian matrix dF/dq
    std::cout << "Jqpx = \n" << Jqpx << std::endl; // print the evaluated Jacobian matrix [dF/dq, dF/dp, dF/dx]
}
