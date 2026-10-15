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
// autodiff include
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    double u;        // the value of the output variable u
    double ux;       // the first order derivative ux computed by autodiff
    double uxexact;  // the exact first order derivative ux
    double uxx;      // the second order derivative uxx computed by autodiff
    double uxxexact; // the exact second order derivative uxx
};

// Evaluate the function and its derivatives (on the host or on the device)
// Note: the expression tree of var lives in host or device memory, so var can not be a kernel parameter: the
// kernel takes the plain value of x and creates the variable itself.
__host__ __device__ Results compute(double xval)
{
    var x = xval;             // the input variable x
    var u = sin(x) * cos(x);  // the output variable u

    auto [ux] = derivativesx(u, wrt(x));    // evaluate the first order derivatives of u
    auto [uxx] = derivativesx(ux, wrt(x));  // evaluate the second order derivatives of ux

    return {val(u), val(ux), val(1 - 2*sin(x)*sin(x)), val(uxx), val(-4*cos(x)*sin(x))};
}

__global__ void kernel(double x, Results* output)
{
    *output = compute(x);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.ux, b.ux) && allclose(a.uxexact, b.uxexact) && allclose(a.uxx, b.uxx) &&
           allclose(a.uxxexact, b.uxxexact);
}

// The derivatives computed by autodiff must also match the exact ones
bool exact(const Results& r)
{
    return examples::allclose(r.ux, r.uxexact) && examples::allclose(r.uxx, r.uxxexact);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;                  // print the evaluated output variable u
    std::cout << "ux(autodiff) = " << r.ux << std::endl;      // print the evaluated first order derivative ux
    std::cout << "ux(exact) = " << r.uxexact << std::endl;    // print the exact first order derivative ux
    std::cout << "uxx(autodiff) = " << r.uxx << std::endl;    // print the evaluated second order derivative uxx
    std::cout << "uxx(exact) = " << r.uxxexact << std::endl;  // print the exact second order derivative uxx
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    double x = 0.5; // the value of the input variable x

    const Results host = compute(x);
    const Results device = examples::runOnDevice<Results>(kernel, x);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc, [](const Results& a, const Results& b) {
        return allclose(a, b) && exact(a) && exact(b);
    });
}
