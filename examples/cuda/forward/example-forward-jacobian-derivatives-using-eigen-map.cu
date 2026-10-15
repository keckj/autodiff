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
using Vector5real = Eigen::Matrix<real, 5, 1>;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
struct Results
{
    Vector5d F;  // the output vector F = f(x)
    double y[25]; // the Jacobian matrix dF/dx as a flat array (column-major)
};

// The vector function for which the Jacobian is needed
__host__ __device__ Vector5real f(const Vector5real& x)
{
    return x * x.sum();
}

// Evaluate the function and its Jacobian (on the host or on the device)
__host__ __device__ Results compute(Vector5real x)
{
    Results results;
    Vector5real F;                          // the output vector F = f(x) evaluated together with Jacobian matrix below
    Eigen::Map<Matrix5d> J(results.y);      // the output Jacobian dF/dx mapped onto the flat array
    jacobian(f, wrt(x), at(x), F, J);       // evaluate the output vector F and the Jacobian matrix dF/dx
    results.F = F.cast<double>();
    return results;
}

__global__ void kernel(Vector5real x, Results* output)
{
    *output = compute(x);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.F, b.F) && allclose(Eigen::Map<const Matrix5d>(a.y), Eigen::Map<const Matrix5d>(b.y));
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "F = \n" << r.F << std::endl;                          // print the evaluated output vector F
    std::cout << "J = \n" << Eigen::Map<const Matrix5d>(r.y) << std::endl; // print the evaluated Jacobian matrix dF/dx
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    Vector5real x;       // the input vector x with 5 variables
    x << 1, 2, 3, 4, 5;  // x = [1, 2, 3, 4, 5]

    const Results host = compute(x);
    const Results device = examples::runOnDevice<Results>(kernel, x);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
