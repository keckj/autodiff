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
#include <autodiff/forward/real.hpp>
using namespace autodiff;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    double uexact;  // the exact value of f(x + t, y + t, z + 2t)
    double utaylor; // the 4th order Taylor estimate of f(x + t, y + t, z + 2t)
};

// The scalar function for which a 4th order directional Taylor series will be computed.
__host__ __device__ real4th f(const real4th& x, const real4th& y, const real4th& z)
{
    return sin(x * y) * cos(x * z) * exp(z);
}

// Evaluate the function and its Taylor approximation (on the host or on the device)
__host__ __device__ Results compute(real4th x, real4th y, real4th z, double t)
{
    auto g = taylorseries(f, along(1, 1, 2), at(x, y, z)); // the function g(t) as a 4th order Taylor approximation of f(x + t, y + t, z + 2t)
    real4th u = f(x + t, y + t, z + 2*t);                  // the exact value of f(x + t, y + t, z + 2t)
    double utaylor = g(t);                                 // the 4th order Taylor estimate of f(x + t, y + t, z + 2t)
    return {val(u), utaylor};
}

__global__ void kernel(real4th x, real4th y, real4th z, double t, Results* output)
{
    *output = compute(x, y, z, t);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.uexact, b.uexact) && allclose(a.utaylor, b.utaylor);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << std::fixed;
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t, y + t, z + 2t):" << std::endl;
    std::cout << "u(exact)  = " << r.uexact << " (expected -16.847071)" << std::endl;
    std::cout << "u(taylor) = " << r.utaylor << " (expected -16.793986)" << std::endl;
    std::cout << std::defaultfloat << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    real4th x = 1.0;  // the input variable x
    real4th y = 2.0;  // the input variable y
    real4th z = 3.0;  // the input variable z
    double t = 0.1;   // the step length used to evaluate g(t), the Taylor approximation of f(x + t, y + t, z + 2t)

    const Results host = compute(x, y, z, t);
    const Results device = examples::runOnDevice<Results>(kernel, x, y, z, t);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, y, z, t);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
