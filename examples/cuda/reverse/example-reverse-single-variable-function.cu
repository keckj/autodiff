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
    double u;  // the value of the output variable u
    double ux; // the derivative du/dx
};

// The single-variable function for which derivatives are needed
__host__ __device__ var f(var x)
{
    return 1 + x + x*x + 1/x + log(x);
}

// Evaluate the function and its derivative (on the host or on the device)
// Note: the expression tree of var lives in host or device memory, so var can not be a kernel parameter: the
// kernel takes the plain value of x and creates the variable itself.
__host__ __device__ Results compute(double xval)
{
    var x = xval;                       // the input variable x
    var u = f(x);                       // the output variable u
    auto [ux] = derivatives(u, wrt(x)); // evaluate the derivative of u with respect to x
    return {val(u), ux};
}

__global__ void kernel(double x, Results* output)
{
    *output = compute(x);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.ux, b.ux);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;    // print the evaluated output variable u
    std::cout << "ux = " << r.ux << std::endl;  // print the evaluated derivative ux
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    double x = 2.0; // the value of the input variable x

    const Results host = compute(x);
    const Results device = examples::runOnDevice<Results>(kernel, x);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
