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
#include <autodiff/reverse/var.hpp>
using namespace autodiff;

// The results computed on the host, on the device with nvcc and on the device with NVRTC
// (stdx::array is cuda::std::array with NVRTC)
struct Results
{
    double u;                                       // the value of the output variable u
    stdx::array<double, 3> du;                      // the first order derivatives [ux, uy, uz]
    stdx::array<stdx::array<double, 3>, 3> d2u;     // the second order derivatives [[uxx, uxy, uxz], [uyx, uyy, uyz], [uzx, uzy, uzz]]
};

// Evaluate the function and its derivatives (on the host or on the device)
// Note: the expression tree of var lives in host or device memory, so var can not be a kernel parameter: the
// kernel takes the plain values of x, y and z and creates the variables itself.
__host__ __device__ Results compute(double xval, double yval, double zval)
{
    var x = xval;  // the input variable x
    var y = yval;  // the input variable y
    var z = zval;  // the input variable z

    var u = x * log(y) * exp(z);  // the output variable u

    auto [ux, uy, uz] = derivativesx(u, wrt(x, y, z));  // evaluate the derivatives of u with respect to x, y, z.

    auto [uxx, uxy, uxz] = derivativesx(ux, wrt(x, y, z)); // evaluate the derivatives of ux with respect to x, y, z.
    auto [uyx, uyy, uyz] = derivativesx(uy, wrt(x, y, z)); // evaluate the derivatives of uy with respect to x, y, z.
    auto [uzx, uzy, uzz] = derivativesx(uz, wrt(x, y, z)); // evaluate the derivatives of uz with respect to x, y, z.

    Results r;
    r.u = val(u);
    r.du = {val(ux), val(uy), val(uz)};
    r.d2u[0] = {val(uxx), val(uxy), val(uxz)};
    r.d2u[1] = {val(uyx), val(uyy), val(uyz)};
    r.d2u[2] = {val(uzx), val(uzy), val(uzz)};
    return r;
}

__global__ void kernel(double x, double y, double z, Results* output)
{
    *output = compute(x, y, z);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.du, b.du) && allclose(a.d2u, b.d2u);
}

void print(const std::string& backend, const Results& r)
{
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;            // print the evaluated output variable u
    std::cout << "ux = " << r.du[0] << std::endl;       // print the evaluated first order derivative ux
    std::cout << "uy = " << r.du[1] << std::endl;       // print the evaluated first order derivative uy
    std::cout << "uz = " << r.du[2] << std::endl;       // print the evaluated first order derivative uz
    std::cout << "uxx = " << r.d2u[0][0] << std::endl;  // print the evaluated second order derivative uxx
    std::cout << "uxy = " << r.d2u[0][1] << std::endl;  // print the evaluated second order derivative uxy
    std::cout << "uxz = " << r.d2u[0][2] << std::endl;  // print the evaluated second order derivative uxz
    std::cout << "uyx = " << r.d2u[1][0] << std::endl;  // print the evaluated second order derivative uyx
    std::cout << "uyy = " << r.d2u[1][1] << std::endl;  // print the evaluated second order derivative uyy
    std::cout << "uyz = " << r.d2u[1][2] << std::endl;  // print the evaluated second order derivative uyz
    std::cout << "uzx = " << r.d2u[2][0] << std::endl;  // print the evaluated second order derivative uzx
    std::cout << "uzy = " << r.d2u[2][1] << std::endl;  // print the evaluated second order derivative uzy
    std::cout << "uzz = " << r.d2u[2][2] << std::endl;  // print the evaluated second order derivative uzz
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    double x = 1.0;  // the value of the input variable x
    double y = 0.5;  // the value of the input variable y
    double z = 2.0;  // the value of the input variable z

    const Results host = compute(x, y, z);
    const Results device = examples::runOnDevice<Results>(kernel, x, y, z);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, y, z);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
