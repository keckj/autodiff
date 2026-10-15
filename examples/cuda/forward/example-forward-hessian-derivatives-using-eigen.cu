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
using Vector3d = Eigen::Matrix<double, 3, 1>;
using Matrix3d = Eigen::Matrix<double, 3, 3>;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    Vector3d g; // the gradient vector g = du/dx
    Matrix3d H; // the Hessian matrix H = d2u/dx2
    double u;   // the output scalar u = f(x)
};

// The scalar function for which the gradient is needed
__host__ __device__ dual2nd f(const Vector3dual2nd& x)
{
    return x.cwiseProduct(x).sum(); // sum([x(i) * x(i) for i = 1:5])
}

// Evaluate the function, its gradient and its Hessian (on the host or on the device)
__host__ __device__ Results compute(Vector3dual2nd x)
{
    dual2nd u;                                    // the output scalar u = f(x) evaluated together with Hessian below
    Vector3dual g;                                // gradient of f(x) evaluated together with Hessian below
    Matrix3d H = hessian(f, wrt(x), at(x), u, g); // evaluate the function value u and its Hessian matrix H
    return {g.cast<double>(), H, val(u)};
}

__global__ void kernel(Vector3dual2nd x, Results* output)
{
    *output = compute(x);
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
    std::cout << "u = "   << r.u << std::endl;  // print the evaluated output u
    std::cout << "g = \n" << r.g << std::endl;  // print the evaluated gradient vector g = du/dx
    std::cout << "H = \n" << r.H << std::endl;  // print the evaluated Hessian matrix H = d2u/dx2
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Vector3dual2nd x;  // the input vector x with 3 variables
    x << 1, 2, 3;      // x = [1, 2, 3]

    const Results host = compute(x);
    const Results device = examples::runOnDevice<Results>(kernel, x);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
