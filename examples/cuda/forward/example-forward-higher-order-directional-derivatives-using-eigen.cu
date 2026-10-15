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
    stdx::array<Array5d, 5> dfdv; // the directional derivatives of f up to 4th order (stdx::array is cuda::std::array with NVRTC)
    Array5d uexact;               // the exact value of f(x + t*vx, p + t*vp)
    Array5d utaylor;              // the 4th order Taylor estimate of f(x + t*vx, p + t*vp)
};

// The vector function for which higher-order directional derivatives are needed (up to 4th order).
__host__ __device__ Array5real4th f(const Array5real4th& x, real4th p)
{
    return p * x.log();
}

// Evaluate the directional derivatives and the Taylor estimate of the function (on the host or on the device)
__host__ __device__ Results compute(Array5real4th x, real4th p, const Array5d& vx, double vp, double t)
{
    auto dfdv = derivatives(f, along(vx, vp), at(x, p)); // the directional derivatives of f along direction v = (vx, vp) at (x, p)

    Array5real4th u = f(x + t * vx, p + t * vp); // start from (x, p), walk a step length t along direction v = (vx, vp) and evaluate f at this current point
    Array5d utaylor = dfdv[0] + t*dfdv[1] + (t*t/2.0)*dfdv[2] + (t*t*t/6.0)*dfdv[3] + (t*t*t*t/24.0)*dfdv[4]; // evaluate the 4th order Taylor estimate of f along direction v = (vx, vp) at a step length of t

    return {dfdv, u.cast<double>(), utaylor};
}

__global__ void kernel(Array5real4th x, real4th p, Array5d vx, double vp, double t, Results* output)
{
    *output = compute(x, p, vx, vp, t);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.dfdv, b.dfdv) && allclose(a.uexact, b.uexact) && allclose(a.utaylor, b.utaylor);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << std::scientific << std::showpos;
    std::cout << "Directional derivatives of f along v = (vx, vp) up to 4th order:" << std::endl;
    std::cout << "dfdv[0] = " << r.dfdv[0].transpose() << std::endl; // print the evaluated 0th order directional derivative of f along v (equivalent to f(x, p))
    std::cout << "dfdv[1] = " << r.dfdv[1].transpose() << std::endl; // print the evaluated 1st order directional derivative of f along v
    std::cout << "dfdv[2] = " << r.dfdv[2].transpose() << std::endl; // print the evaluated 2nd order directional derivative of f along v
    std::cout << "dfdv[3] = " << r.dfdv[3].transpose() << std::endl; // print the evaluated 3rd order directional derivative of f along v
    std::cout << "dfdv[4] = " << r.dfdv[4].transpose() << std::endl; // print the evaluated 4th order directional derivative of f along v
    std::cout << "Comparison between exact evaluation and 4th order Taylor estimate:" << std::endl;
    std::cout << "u(exact)  = " << r.uexact.transpose() << std::endl;
    std::cout << "u(taylor) = " << r.utaylor.transpose() << std::endl;
    std::cout << std::defaultfloat << std::noshowpos << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Array5real4th x;   // the input vector x
    x << 1.0, 2.0, 3.0, 4.0, 5.0;

    real4th p = 7.0;   // the input parameter p = 7

    Array5d vx;        // the direction vx in the direction vector v = (vx, vp)
    vx << 1.0, 1.0, 1.0, 1.0, 1.0;

    double vp = 1.0;   // the direction vp in the direction vector v = (vx, vp)

    double t = 0.1;    // the step length along direction v = (vx, vp) used to compute 4th order Taylor estimate of f

    const Results host = compute(x, p, vx, vp, t);
    const Results device = examples::runOnDevice<Results>(kernel, x, p, vx, vp, t);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, p, vx, vp, t);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
