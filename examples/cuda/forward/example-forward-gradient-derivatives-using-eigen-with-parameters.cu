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
using Vector1d = Eigen::Matrix<double, 1, 1>;
using Vector3d = Eigen::Matrix<double, 3, 1>;
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Vector9d = Eigen::Matrix<double, 9, 1>;
using Array3real = Eigen::Array<real, 3, 1>;
using Array5real = Eigen::Array<real, 5, 1>;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    Vector5d gx;   // the gradient vector gx = du/dx
    Vector3d gp;   // the gradient vector gp = du/dp
    Vector1d gq;   // the gradient vector gq = du/dq
    Vector9d gqpx; // the gradient vector gqpx = [du/dq, du/dp, du/dx]
    double u;      // the output scalar u = f(x, p, q)
};

// The scalar function for which the gradient is needed
__host__ __device__ real f(const Array5real& x, const Array3real& p, const real& q)
{
    return (x * x).sum() * p.sum() * exp(q); // sum([xi * xi for i = 1:5]) * sum(p) * exp(q)
}

// Evaluate the function and its gradients (on the host or on the device)
__host__ __device__ Results compute(Array5real x, Array3real p, real q)
{
    real u; // the output scalar u = f(x, p, q) evaluated together with gradient below

    Vector5d gx   = gradient(f, wrt(x), at(x, p, q), u);       // evaluate the function value u and its gradient vector gx = du/dx
    Vector3d gp   = gradient(f, wrt(p), at(x, p, q), u);       // evaluate the function value u and its gradient vector gp = du/dp
    Vector1d gq   = gradient(f, wrt(q), at(x, p, q), u);       // evaluate the function value u and its gradient vector gq = du/dq
    Vector9d gqpx = gradient(f, wrt(q, p, x), at(x, p, q), u); // evaluate the function value u and its gradient vector gqpx = [du/dq, du/dp, du/dx]

    return {gx, gp, gq, gqpx, val(u)};
}

__global__ void kernel(Array5real x, Array3real p, real q, Results* output)
{
    *output = compute(x, p, q);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.gx, b.gx) && allclose(a.gp, b.gp) && allclose(a.gq, b.gq) &&
           allclose(a.gqpx, b.gqpx);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;          // print the evaluated output u
    std::cout << "gx = \n" << r.gx << std::endl;      // print the evaluated gradient vector gx = du/dx
    std::cout << "gp = \n" << r.gp << std::endl;      // print the evaluated gradient vector gp = du/dp
    std::cout << "gq = \n" << r.gq << std::endl;      // print the evaluated gradient vector gq = du/dq
    std::cout << "gqpx = \n" << r.gqpx << std::endl;  // print the evaluated gradient vector gqpx = [du/dq, du/dp, du/dx]
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Array5real x;        // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    Array3real p;        // the input parameter vector p with 3 variables
    p << 1, 2, 3;        // p = [1, 2, 3]

    real q = -2;         // the input parameter q as a single variable

    const Results host = compute(x, p, q);
    const Results device = examples::runOnDevice<Results>(kernel, x, p, q);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, p, q);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
