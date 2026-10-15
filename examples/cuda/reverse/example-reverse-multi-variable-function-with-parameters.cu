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
    double ua; // the derivative du/da
    double ub; // the derivative du/db
    double uc; // the derivative du/dc
};

// A type defining parameters for a function of interest
struct Params
{
    var a;
    var b;
    var c;
};

// The values of the parameters, passed to the kernels (var can not be a kernel parameter, see compute)
struct ParamValues
{
    double a;
    double b;
    double c;
};

// The function that depends on parameters for which derivatives are needed
__host__ __device__ var f(var x, const Params& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

// Evaluate the function and its derivatives (on the host or on the device)
// Note: the expression tree of var lives in host or device memory, so var can not be a kernel parameter: the
// kernel takes the plain values of x and of the parameters and creates the variables itself.
__host__ __device__ Results compute(double xval, ParamValues pvals)
{
    Params params;        // initialize the parameter variables
    params.a = pvals.a;   // the parameter a of type var, not double!
    params.b = pvals.b;   // the parameter b of type var, not double!
    params.c = pvals.c;   // the parameter c of type var, not double!

    var x = xval;         // the input variable x
    var u = f(x, params); // the output variable u

    auto [ux, ua, ub, uc] = derivatives(u, wrt(x, params.a, params.b, params.c)); // evaluate the derivatives of u with respect to x and parameters a, b, c

    return {val(u), ux, ua, ub, uc};
}

__global__ void kernel(double x, ParamValues params, Results* output)
{
    *output = compute(x, params);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.ux, b.ux) && allclose(a.ua, b.ua) && allclose(a.ub, b.ub) &&
           allclose(a.uc, b.uc);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;    // print the evaluated output u
    std::cout << "ux = " << r.ux << std::endl;  // print the evaluated derivative du/dx
    std::cout << "ua = " << r.ua << std::endl;  // print the evaluated derivative du/da
    std::cout << "ub = " << r.ub << std::endl;  // print the evaluated derivative du/db
    std::cout << "uc = " << r.uc << std::endl;  // print the evaluated derivative du/dc
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    ParamValues params; // the values of the parameters
    params.a = 1.0;
    params.b = 2.0;
    params.c = 3.0;

    double x = 0.5;     // the value of the input variable x

    const Results host = compute(x, params);
    const Results device = examples::runOnDevice<Results>(kernel, x, params);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, params);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
