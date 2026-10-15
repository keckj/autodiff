//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// CUDA examples harness
#include "common.hpp"

AUTODIFF_NVRTC_BEGIN_SOURCE
// libcu++ include (std::complex is not usable in device code, cuda::std::complex is)
#include <cuda/std/complex>

// autodiff include
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// Specialize isArithmetic for complex to make it compatible with dual
namespace autodiff::detail {

template<typename T>
struct ArithmeticTraits<cuda::std::complex<T>> : ArithmeticTraits<T> {};

} // autodiff::detail

using complex = cuda::std::complex<double>;
using cxdual = Dual<complex, complex>;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    complex u;    // the value of the output variable u
    complex dudx; // the derivative du/dx
};

// The single-variable function for which derivatives are needed
__host__ __device__ cxdual f(cxdual x)
{
    return 1 + x + x*x + 1/x + log(x);
}

// Evaluate the function and its derivative (on the host or on the device)
__host__ __device__ Results compute(cxdual x)
{
    cxdual u = f(x);                              // the output variable u
    cxdual dudx = derivative(f, wrt(x), at(x));   // evaluate the derivative du/dx
    return {u.val, dudx.val};
}

__global__ void kernel(cxdual x, Results* output)
{
    *output = compute(x);
}
AUTODIFF_NVRTC_END_SOURCE

std::string str(const complex& z)
{
    std::ostringstream ss;
    ss << "(" << z.real() << "," << z.imag() << ")";
    return ss.str();
}

bool allclose(const complex& a, const complex& b)
{
    return examples::allclose(a.real(), b.real()) && examples::allclose(a.imag(), b.imag());
}

bool allclose(const Results& a, const Results& b)
{
    return allclose(a.u, b.u) && allclose(a.dudx, b.dudx);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << str(r.u) << std::endl;         // print the evaluated output u
    std::cout << "du/dx = " << str(r.dudx) << std::endl;  // print the evaluated derivative du/dx
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    cxdual x = 2.0; // the input variable x

    const Results host = compute(x);
    const Results device = examples::runOnDevice<Results>(kernel, x);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
