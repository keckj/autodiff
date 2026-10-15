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
#include <autodiff/forward/real.hpp>
using namespace autodiff;

// The results computed on the host, on the device with nvcc and on the device with NVRTC:
// the directional derivatives dfdv returned by derivatives (stdx::array is cuda::std::array with NVRTC)
using Results = stdx::array<double, 5>;

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
__host__ __device__ real4th f(real4th x, real4th y, real4th z)
{
    return sin(x) * cos(y) * exp(z);
}

// Evaluate the directional derivatives of the function (on the host or on the device)
__host__ __device__ Results compute(real4th x, real4th y, real4th z)
{
    return derivatives(f, along(1, 1, 2), at(x, y, z)); // the directional derivatives of f along direction v = (1, 1, 2) at (x, y, z) = (1, 2, 3)
}

__global__ void kernel(real4th x, real4th y, real4th z, Results* output)
{
    *output = compute(x, y, z);
}
AUTODIFF_NVRTC_END_SOURCE

void print(const std::string& backend, const Results& dfdv)
{
    std::cout << backend << std::endl;
    std::cout << "dfdv[0] = " << dfdv[0] << " (expected -7.03346)" << std::endl; // print the evaluated 0th order directional derivative of f along v (equivalent to f(x, y, z))
    std::cout << "dfdv[1] = " << dfdv[1] << " (expected -33.9515)" << std::endl; // print the evaluated 1st order directional derivative of f along v
    std::cout << "dfdv[2] = " << dfdv[2] << " (expected -113.341)" << std::endl; // print the evaluated 2nd order directional derivative of f along v
    std::cout << "dfdv[3] = " << dfdv[3] << " (expected -249.358)" << std::endl; // print the evaluated 3rd order directional derivative of f along v
    std::cout << "dfdv[4] = " << dfdv[4] << " (expected -225.914)" << std::endl; // print the evaluated 4th order directional derivative of f along v
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    real4th x = 1.0;
    real4th y = 2.0;
    real4th z = 3.0;

    const Results host = compute(x, y, z);
    const Results device = examples::runOnDevice<Results>(kernel, x, y, z);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, y, z);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
