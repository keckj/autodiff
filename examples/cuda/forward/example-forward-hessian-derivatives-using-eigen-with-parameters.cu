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
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

// Fixed size vectors and matrices (dynamic size ones would be allocated on the device heap)
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6dual = Eigen::Matrix<dual, 6, 1>;
using Array2dual2nd = Eigen::Array<dual2nd, 2, 1>;
using Array3dual2nd = Eigen::Array<dual2nd, 3, 1>;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    Vector6d g; // the gradient vector g = [du/dx, du/dp, du/dq]
    Matrix6d H; // the Hessian matrix H = d2u/d[x, p, q]2
    double u;   // the output scalar u = f(x, p, q)
};

// The scalar function for which the Hessian is needed
__host__ __device__ dual2nd f(const Array3dual2nd& x, const Array2dual2nd& p, const dual2nd& q)
{
    return (x * x).sum() + (p * p).sum() * q; // sum(x*x) + sum(p*p) + q
}

// Evaluate the function, its gradient and its Hessian (on the host or on the device)
__host__ __device__ Results compute(Array3dual2nd x, Array2dual2nd p, dual2nd q)
{
    dual2nd u;     // the output scalar u = f(x) evaluated together with Hessian below
    Vector6dual g; // gradient of f(x) evaluated together with Hessian below

    Matrix6d H = hessian(f, wrt(x, p, q), at(x, p, q), u, g); // evaluate the function value u, its gradient vector g, and its Hessian matrix H with respect to (x, p, q)

    return {g.cast<double>(), H, val(u)};
}

__global__ void kernel(Array3dual2nd x, Array2dual2nd p, dual2nd q, Results* output)
{
    *output = compute(x, p, q);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.g, b.g) && allclose(a.H, b.H);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;    // print the evaluated output u
    std::cout << "g =\n" << r.g << std::endl;   // print the evaluated gradient vector g = [du/dx, du/dp, du/dq]
    std::cout << "H = \n" << r.H << std::endl;  // print the evaluated Hessian matrix H = d2u/d[x, p, q]2
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Array3dual2nd x;  // the input vector x with 3 variables
    x << 1, 2, 3;

    Array2dual2nd p;  // the input parameter vector p with 2 variables
    p << 4, 5;

    dual2nd q = -2;   // the input parameter q as a single variable

    const Results host = compute(x, p, q);
    const Results device = examples::runOnDevice<Results>(kernel, x, p, q);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, p, q);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
