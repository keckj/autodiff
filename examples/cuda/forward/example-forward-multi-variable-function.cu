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
    double dudy; // the derivative du/dy
    double dudz; // the derivative du/dz
};

// The multi-variable function for which derivatives are needed
__host__ __device__ dual f(dual x, dual y, dual z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

// Evaluate the function and its derivatives (on the host or on the device)
__host__ __device__ Results compute(dual x, dual y, dual z)
{
    dual u = f(x, y, z);
    double dudx = derivative(f, wrt(x), at(x, y, z));
    double dudy = derivative(f, wrt(y), at(x, y, z));
    double dudz = derivative(f, wrt(z), at(x, y, z));
    return {val(u), dudx, dudy, dudz};
}

__global__ void kernel(dual x, dual y, dual z, Results* output)
{
    *output = compute(x, y, z);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.dudx, b.dudx) && allclose(a.dudy, b.dudy) && allclose(a.dudz, b.dudz);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;         // print the evaluated output u = f(x, y, z)
    std::cout << "du/dx = " << r.dudx << std::endl;  // print the evaluated derivative du/dx
    std::cout << "du/dy = " << r.dudy << std::endl;  // print the evaluated derivative du/dy
    std::cout << "du/dz = " << r.dudz << std::endl;  // print the evaluated derivative du/dz
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    dual x = 1.0;
    dual y = 2.0;
    dual z = 3.0;

    const Results host = compute(x, y, z);
    const Results device = examples::runOnDevice<Results>(kernel, x, y, z);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, y, z);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
