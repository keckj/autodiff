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
    double x0, y0, u0, ux0, uy0; // the inputs, the output u and its derivatives before updating x
    double x1, y1, u1, ux1, uy1; // the inputs, the output u and its derivatives after updating x
    double minxy;                // min(x, y) after updating x
    double maxxy;                // max(x, y) after updating x
    double sgnx;                 // sgn(x) after updating x
};

// A two-variable piecewise function for which derivatives are needed
__host__ __device__ var f(var x, var y) { return condition(x < y, x * y, x * x); }

// Evaluate the function and its derivatives, update x and evaluate them again (on the host or on the device)
// Note: the expression tree of var lives in host or device memory, so var can not be a kernel parameter: the
// kernel takes the plain values of x and y and creates the variables itself.
__host__ __device__ Results compute(double xval, double yval, double xnew)
{
    Results r;

    var x = xval;     // the input variable x
    var y = yval;     // the input variable y
    var u = f(x, y);  // the output variable u

    auto [ux, uy] = derivatives(u, wrt(x, y)); // evaluate the derivatives of u
    r.x0 = val(x);
    r.y0 = val(y);
    r.u0 = val(u);
    r.ux0 = ux;
    r.uy0 = uy;

    x.update(xnew); // Change the value of x in the expression tree
    u.update();     // Update the expression tree in a sweep

    auto [ux2, uy2] = derivatives(u, wrt(x, y)); // re-evaluate the derivatives
    r.x1 = val(x);
    r.y1 = val(y);
    r.u1 = val(u);
    r.ux1 = ux2;
    r.uy1 = uy2;

    // condition-associated functions
    r.minxy = val(min(x, y));
    r.maxxy = val(max(x, y));
    r.sgnx = val(sgn(x));

    return r;
}

__global__ void kernel(double x, double y, double xnew, Results* output)
{
    *output = compute(x, y, xnew);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.x0, b.x0) && allclose(a.y0, b.y0) && allclose(a.u0, b.u0) && allclose(a.ux0, b.ux0) &&
           allclose(a.uy0, b.uy0) && allclose(a.x1, b.x1) && allclose(a.y1, b.y1) && allclose(a.u1, b.u1) &&
           allclose(a.ux1, b.ux1) && allclose(a.uy1, b.uy1) && allclose(a.minxy, b.minxy) &&
           allclose(a.maxxy, b.maxxy) && allclose(a.sgnx, b.sgnx);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "x = " << r.x0 << ", y = " << r.y0 << std::endl;
    std::cout << "u = " << r.u0 << " (expected 2)" << std::endl;    // x = 1, y = 2, so x < y, so x * y = 2
    std::cout << "ux = " << r.ux0 << " (expected 2)" << std::endl;  // d/dx x * y = y = 2
    std::cout << "uy = " << r.uy0 << " (expected 1)" << std::endl;  // d/dy x * y = x = 1
    std::cout << "x = " << r.x1 << ", y = " << r.y1 << std::endl;
    std::cout << "u = " << r.u1 << " (expected 9)" << std::endl;    // Now x > y, so x * x = 9
    std::cout << "ux = " << r.ux1 << " (expected 6)" << std::endl;  // d/dx x * x = 2x = 6
    std::cout << "uy = " << r.uy1 << " (expected 0)" << std::endl;  // d/dy x * x = 0
    std::cout << "min(x, y) = " << r.minxy << " (expected 2)" << std::endl;
    std::cout << "max(x, y) = " << r.maxxy << " (expected 3)" << std::endl;
    std::cout << "sgn(x) = " << r.sgnx << " (expected 1)" << std::endl;
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    double x = 1.0;    // the initial value of the input variable x
    double y = 2.0;    // the value of the input variable y
    double xnew = 3.0; // the updated value of the input variable x

    const Results host = compute(x, y, xnew);
    const Results device = examples::runOnDevice<Results>(kernel, x, y, xnew);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, y, xnew);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
