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
struct Results
{
    double u;                   // the value of the output variable u
    stdx::array<double, 3> du;  // the derivatives [ux, uy, uz] (stdx::array is cuda::std::array with NVRTC)
};

// The multi-variable function for which derivatives are needed
__host__ __device__ var f(var x, var y, var z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

// Evaluate the function and its derivatives (on the host or on the device)
// Note: var can not be a kernel parameter, a kernel like
//     __global__ void kernel(var x, var y, var z, Results* output);
// would be undefined behavior: the expression tree of a var created on the host lives in host memory (and its nodes
// have virtual functions whose table is only valid on the host). The kernel takes plain values and creates the
// variables itself.
__host__ __device__ Results compute(double xval, double yval, double zval)
{
    var x = xval;                                       // the input variable x
    var y = yval;                                       // the input variable y
    var z = zval;                                       // the input variable z
    var u = f(x, y, z);                                 // the output variable u
    stdx::array<double, 3> du = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z
    return {val(u), du};
}

__global__ void kernel(double x, double y, double z, Results* output)
{
    *output = compute(x, y, z);
}
AUTODIFF_NVRTC_END_SOURCE

bool allclose(const Results& a, const Results& b)
{
    using examples::allclose;
    return allclose(a.u, b.u) && allclose(a.du, b.du);
}

void print(const std::string& backend, const Results& r)
{
    const auto& [ux, uy, uz] = r.du;
    std::cout << backend << std::endl;
    std::cout << "u = " << r.u << std::endl;  // print the evaluated output u
    std::cout << "ux = " << ux << std::endl;  // print the evaluated derivative ux
    std::cout << "uy = " << uy << std::endl;  // print the evaluated derivative uy
    std::cout << "uz = " << uz << std::endl;  // print the evaluated derivative uz
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    double x = 1.0; // the value of the input variable x
    double y = 2.0; // the value of the input variable y
    double z = 3.0; // the value of the input variable z

    const Results host = compute(x, y, z);
    const Results device = examples::runOnDevice<Results>(kernel, x, y, z);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, y, z);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
