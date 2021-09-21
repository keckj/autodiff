// C++ includes
#include <iostream>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

Array5real4th f(const Array5real4th& x)
{
    return x.sin() / x;
}

int main()
{
    Array<real4th,5> x(1.0, 2.0, 3.0, 4.0, 5.0);

    Array<double,5> v(1.0, 1.0, 1.0, 1.0, 1.0);

    auto g = taylorseries(f, along(v), at(x)); // the function g(t) as a 4th order Taylor approximation of f(x + t·v)

    double t = 0.1;                            // the step length used to evaluate g(t), the Taylor approximation of f(x + t·v)

    Array5real4th u = f(x + t * v);            // the exact value of f(x + t·v)

    Array5real utaylor = g(t);                 // the 4th order Taylor estimate of f(x + t·v)

    std::cout << std::fixed;
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t·v):" << std::endl;
    std::cout << "u(exact)  = " << u.transpose() << std::endl;
    std::cout << "u(taylor) = " << utaylor.transpose() << std::endl;
}

/*-------------------------------------------------------------------------------------------------
=== Output ===
---------------------------------------------------------------------------------------------------
Comparison between exact evaluation and 4th order Taylor estimate of f(x + t·v):
u(exact)  =  0.810189  0.411052  0.013413 -0.199580 -0.181532
u(taylor) =  0.810189  0.411052  0.013413 -0.199580 -0.181532
-------------------------------------------------------------------------------------------------*/
