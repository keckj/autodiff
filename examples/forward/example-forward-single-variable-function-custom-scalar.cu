// C++ includes
#include <iostream>
#include <cuda/std/complex>

template<typename T>
__host__ std::ostream& operator<<(std::ostream& os, const cuda::std::complex<T>& v)
{
    os << v.real() << " + " << v.imag() << "i";
    return os;
}

// autodiff include
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// Specialize isArithmetic for complex to make it compatible with dual
namespace autodiff::detail {

template<typename T>
struct ArithmeticTraits<cuda::std::complex<T>> : ArithmeticTraits<T> {};

} // autodiff::detail

using cxdual = Dual<cuda::std::complex<double>, cuda::std::complex<double>>;


// The single-variable function for which derivatives are needed
cxdual f(cxdual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

int main()
{
    cxdual x = cuda::std::complex<double>(2.0,1.0);   // the input variable x
    cxdual u = f(x);  // the output variable u

    cxdual dudx = derivative(f, wrt(x), at(x));  // evaluate the derivative du/dx

    std::cout << "u = " << u.val << " (expected 7.20472 + 5.26365i)" << std::endl;         // print the evaluated output u
    std::cout << "du/dx = " << dudx.val << " (expected 5.28 + 1.96i)" << std::endl;  // print the evaluated derivative du/dx
}
