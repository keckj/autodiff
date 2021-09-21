// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The vector function for which the Jacobian is needed
Vector<real,5> f(const Vector<real,5>& x)
{
    return x * x.sum();
}

int main()
{
    Vector<real,5> x(1, 2, 3, 4, 5);

    Vector<real,5> F;                          // the output vector F = f(x) evaluated together with Jacobian matrix below

    Matrix<double,5,5> J = jacobian(f, wrt(x), at(x), F); // evaluate the output vector F and the Jacobian matrix dF/dx

    std::cout << "F = \n" << F << std::endl;    // print the evaluated output vector F
    std::cout << "J = \n" << J << std::endl;    // print the evaluated Jacobian matrix dF/dx
}
