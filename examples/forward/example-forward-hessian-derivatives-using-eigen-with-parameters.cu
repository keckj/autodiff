// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

// The scalar function for which the Hessian is needed
dual2nd f(const Array<dual2nd,3>& x, const Array<dual2nd,2>& p, const dual2nd& q)
{
    return (x * x).sum() + (p * p).sum() * q; // sum(x*x) + sum(p*p) + q
}

int main()
{
    Array<dual2nd,3> x(1, 2, 3);

    Array<dual2nd,2> p(4, 5);

    dual2nd q = -2;

    dual2nd u; // the output scalar u = f(x) evaluated together with Hessian below

    Vector<dual,6> g; // gradient of f(x) evaluated together with Hessian below

    Matrix<double,6,6> H = hessian(f, wrt(x, p, q), at(x, p, q), u, g); // evaluate the function value u, its gradient vector g, and its Hessian matrix H with respect to (x, p, q)

    std::cout << "u = " << u << std::endl; // print the evaluated output u
    std::cout << "g =\n" << g << std::endl; // print the evaluated gradient vector g = [du/dx, du/dp, du/dq]
    std::cout << "H = \n" << H << std::endl; // print the evaluated Hessian matrix H = d²u/d[x, p, q]²
}
