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

// Fixed size vectors and matrices (dynamic size ones would be allocated on the device heap)
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Matrix5d = Eigen::Matrix<double, 5, 5>;
using Matrix53d = Eigen::Matrix<double, 5, 3>;
using Matrix51d = Eigen::Matrix<double, 5, 1>;
using Matrix59d = Eigen::Matrix<double, 5, 9>;
using Vector3real = Eigen::Matrix<real, 3, 1>;
using Vector5real = Eigen::Matrix<real, 5, 1>;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    Vector5d F;     // the output vector F = f(x, p, q)
    Matrix5d Jx;    // the Jacobian matrix Jx = dF/dx
    Matrix53d Jp;   // the Jacobian matrix Jp = dF/dp
    Matrix51d Jq;   // the Jacobian matrix Jq = dF/dq
    Matrix59d Jqpx; // the Jacobian matrix Jqpx = [dF/dq, dF/dp, dF/dx]
};

// The vector function with parameters for which the Jacobian is needed
__host__ __device__ Vector5real f(const Vector5real& x, const Vector3real& p, const real& q)
{
    return x * p.sum() * exp(q);
}

// Evaluate the function and its Jacobians (on the host or on the device)
__host__ __device__ Results compute(Vector5real x, Vector3real p, real q)
{
    Vector5real F; // the output vector F = f(x, p, q) evaluated together with Jacobian below

    Matrix5d Jx   = jacobian(f, wrt(x), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jx = dF/dx
    Matrix53d Jp  = jacobian(f, wrt(p), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jp = dF/dp
    Matrix51d Jq  = jacobian(f, wrt(q), at(x, p, q), F);       // evaluate the function and the Jacobian matrix Jq = dF/dq
    Matrix59d Jqpx = jacobian(f, wrt(q, p, x), at(x, p, q), F); // evaluate the function and the Jacobian matrix Jqpx = [dF/dq, dF/dp, dF/dx]

    return {F.cast<double>(), Jx, Jp, Jq, Jqpx};
}

__global__ void kernel(Vector5real x, Vector3real p, real q, Results* output)
{
    *output = compute(x, p, q);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.F, b.F) && allclose(a.Jx, b.Jx) && allclose(a.Jp, b.Jp) && allclose(a.Jq, b.Jq) &&
           allclose(a.Jqpx, b.Jqpx);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "F = \n" << r.F << std::endl;        // print the evaluated output vector F
    std::cout << "Jx = \n" << r.Jx << std::endl;      // print the evaluated Jacobian matrix dF/dx
    std::cout << "Jp = \n" << r.Jp << std::endl;      // print the evaluated Jacobian matrix dF/dp
    std::cout << "Jq = \n" << r.Jq << std::endl;      // print the evaluated Jacobian matrix dF/dq
    std::cout << "Jqpx = \n" << r.Jqpx << std::endl;  // print the evaluated Jacobian matrix [dF/dq, dF/dp, dF/dx]
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Vector5real x;       // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    Vector3real p;       // the input parameter vector p with 3 variables
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
