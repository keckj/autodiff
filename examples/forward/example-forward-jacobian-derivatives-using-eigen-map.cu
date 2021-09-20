// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// The vector function for which the Jacobian is needed
VectorX<real,5> f(const VectorX<real,5>& x)
{
    return x * x.sum();
}

int main()
{

    VectorX<real,5> x(1, 2, 3, 4, 5);           // x = [1, 2, 3, 4, 5]
    double y[25];                               // the output Jacobian as a flat array

    VectorX<real,5> F;                          // the output vector F = f(x) evaluated together with Jacobian matrix below
    Eigen::Map<MatrixX<double,5,5>> J(y);       // the output Jacobian dF/dx mapped onto the flat array

    jacobian(f, wrt(x), at(x), F, J);           // evaluate the output vector F and the Jacobian matrix dF/dx

    std::cout << "F = \n" << F << std::endl;    // print the evaluated output vector F
    std::cout << "J = \n" << J << std::endl;    // print the evaluated Jacobian matrix dF/dx
    std::cout << "y = ";                        // print the flat array
    for(int i = 0 ; i < 25 ; ++i)
        std::cout << y[i] << " ";
    std::cout << std::endl;
}
