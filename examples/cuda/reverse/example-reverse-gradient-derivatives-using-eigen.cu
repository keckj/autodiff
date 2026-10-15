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
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

// Fixed size vectors (dynamic size vectors would be allocated on the device heap)
using Vector5d = Eigen::Matrix<double, 5, 1>;
using Vector5var = Eigen::Matrix<var, 5, 1>;
using Array5var = Eigen::Array<var, 5, 1>;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    Vector5d dydx; // the gradient vector dy/dx
    double y;      // the value of the output variable y
};

// The scalar function for which the gradient is needed
__host__ __device__ var f(const Array5var& x)
{
    return sqrt((x * x).sum()); // sqrt(sum([xi * xi for i = 1:5]))
}

// Evaluate the function and its gradient (on the host or on the device)
// Note: the expression tree of var lives in host or device memory, so var can not be a kernel parameter: the
// kernel takes the plain values of x and creates the variables itself.
__host__ __device__ Results compute(const Vector5d& xvals)
{
    Vector5var x = xvals.cast<var>(); // the input vector x with 5 variables
    var y = f(x);                     // the output variable y
    Vector5d dydx = gradient(y, x);   // evaluate the gradient vector dy/dx
    return {dydx, val(y)};
}

__global__ void kernel(Vector5d x, Results* output)
{
    *output = compute(x);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.y, b.y) && allclose(a.dydx, b.dydx);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "y = " << r.y << std::endl;            // print the evaluated output y
    std::cout << "dy/dx = \n" << r.dydx << std::endl;   // print the evaluated gradient vector dy/dx
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Vector5d x;          // the values of the input vector x
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    const Results host = compute(x);
    const Results device = examples::runOnDevice<Results>(kernel, x);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
