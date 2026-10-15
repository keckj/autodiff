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
// Eigen must be included first with NVRTC, before any header using std::
#include <Eigen/Core>

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;

// Fixed size vectors (dynamic size vectors would be allocated on the device heap)
using Array5d = Eigen::Array<double, 5, 1>;
using Array5real4th = Eigen::Array<real4th, 5, 1>;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    Array5d uexact;  // the exact value of f(x + t*v)
    Array5d utaylor; // the 4th order Taylor estimate of f(x + t*v)
};

// The vector function for which a 4th order directional Taylor series will be computed.
__host__ __device__ Array5real4th f(const Array5real4th& x)
{
    return x.sin() / x;
}

// Evaluate the function and its Taylor approximation (on the host or on the device)
// Note: x is taken by value, not by const reference, since taylorseries seeds (modifies) the arguments given to at
__host__ __device__ Results compute(Array5real4th x, const Array5d& v, double t)
{
    auto g = taylorseries(f, along(v), at(x)); // the function g(t) as a 4th order Taylor approximation of f(x + t*v)
    Array5real4th u = f(x + t * v);            // the exact value of f(x + t*v)
    Array5d utaylor = g(t);                    // the 4th order Taylor estimate of f(x + t*v)
    return {u.cast<double>(), utaylor};
}

__global__ void kernel(Array5real4th x, Array5d v, double t, Results* output)
{
    *output = compute(x, v, t);
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
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate of f(x + t*v):" << std::endl;
    std::cout << "u(exact)  = " << r.uexact.transpose() << std::endl;
    std::cout << "u(taylor) = " << r.utaylor.transpose() << std::endl;
    std::cout << std::defaultfloat << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Array5real4th x;  // the input vector x
    x << 1.0, 2.0, 3.0, 4.0, 5.0;
    Array5d v;        // the direction vector v
    v << 1.0, 1.0, 1.0, 1.0, 1.0;
    double t = 0.1;   // the step length used to evaluate g(t), the Taylor approximation of f(x + t*v)

    const Results host = compute(x, v, t);
    const Results device = examples::runOnDevice<Results>(kernel, x, v, t);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, v, t);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
