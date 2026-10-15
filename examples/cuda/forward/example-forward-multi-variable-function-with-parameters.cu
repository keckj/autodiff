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
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    double u;    // the value of the output variable u
    double dudx; // the derivative du/dx
    double duda; // the derivative du/da
    double dudb; // the derivative du/db
    double dudc; // the derivative du/dc
};

// A type defining parameters for a function of interest
struct Params
{
    dual a;
    dual b;
    dual c;
};

// The function that depends on parameters for which derivatives are needed
__host__ __device__ dual f(dual x, const Params& params)
{
    return params.a * sin(x) + params.b * cos(x) + params.c * sin(x)*cos(x);
}

// Evaluate the function and its derivatives (on the host or on the device)
__host__ __device__ Results compute(dual x, Params params)
{
    dual u = f(x, params);                                     // the output variable u
    double dudx = derivative(f, wrt(x), at(x, params));        // evaluate the derivative du/dx
    double duda = derivative(f, wrt(params.a), at(x, params)); // evaluate the derivative du/da
    double dudb = derivative(f, wrt(params.b), at(x, params)); // evaluate the derivative du/db
    double dudc = derivative(f, wrt(params.c), at(x, params)); // evaluate the derivative du/dc
    return {val(u), dudx, duda, dudb, dudc};
}

__global__ void kernel(dual x, Params params, Results* output)
{
    *output = compute(x, params);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.dudx, b.dudx) && allclose(a.duda, b.duda) && allclose(a.dudb, b.dudb) &&
           allclose(a.dudc, b.dudc);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;         // print the evaluated output u
    std::cout << "du/dx = " << r.dudx << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/da = " << r.duda << std::endl;  // print the evaluated derivative du/da
    std::cout << "du/db = " << r.dudb << std::endl;  // print the evaluated derivative du/db
    std::cout << "du/dc = " << r.dudc << std::endl;  // print the evaluated derivative du/dc
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Params params;   // initialize the parameter variables
    params.a = 1.0;  // the parameter a of type dual, not double!
    params.b = 2.0;  // the parameter b of type dual, not double!
    params.c = 3.0;  // the parameter c of type dual, not double!

    dual x = 0.5;  // the input variable x

    const Results host = compute(x, params);
    const Results device = examples::runOnDevice<Results>(kernel, x, params);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, params);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
