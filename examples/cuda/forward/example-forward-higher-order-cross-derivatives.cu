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
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

// The results computed on the host, on the device with nvcc and on the device with NVRTC:
// the array [u0, ux, uxy, uxyx, uxyxz] returned by derivatives (stdx::array is cuda::std::array with NVRTC)
using Results = stdx::array<double, 5>;

// The multi-variable function for which higher-order derivatives are needed (up to 4th order)
__host__ __device__ dual4th f(dual4th x, dual4th y, dual4th z)
{
    return 1 + x + y + z + x*y + y*z + x*z + x*y*z + exp(x/y + y/z);
}

// Evaluate the function and its derivatives (on the host or on the device)
__host__ __device__ Results compute(dual4th x, dual4th y, dual4th z)
{
    return derivatives(f, wrt(x, y, x, z), at(x, y, z));
}

__global__ void kernel(dual4th x, dual4th y, dual4th z, Results* output)
{
    *output = compute(x, y, z);
}
AUTODIFF_NVRTC_END_SOURCE

void print(const std::string& backend, const Results& r)
{
    const auto& [u0, ux, uxy, uxyx, uxyxz] = r;
    std::cout << backend << std::endl;
    std::cout << "u0 = " << u0 << " (expected 27.2113)" << std::endl;         // print the evaluated value of u = f(x, y, z)
    std::cout << "ux = " << ux << " (expected 13.6056)" << std::endl;         // print the evaluated derivative du/dx
    std::cout << "uxy = " << uxy << " (expected 3.33099)" << std::endl;       // print the evaluated derivative d2u/dxdy
    std::cout << "uxyx = " << uxyx << " (expected -0.735916)" << std::endl;   // print the evaluated derivative d3u/dx2dy
    std::cout << "uxyxz = " << uxyxz << " (expected 0.074335)" << std::endl;  // print the evaluated derivative d4u/dx2dydz
    std::cout << std::endl;
}

int main()
{
    examples::printHeader(__FILE__);

    dual4th x = 1.0;
    dual4th y = 2.0;
    dual4th z = 3.0;

    const Results host = compute(x, y, z);
    const Results device = examples::runOnDevice<Results>(kernel, x, y, z);
    const Results nvrtc = examples::runWithNvrtc<Results>(__FILE__, "kernel", x, y, z);

    print("HOST", host);
    print("DEVICE", device);
    print("NVRTC", nvrtc);

    return examples::exitExample(host, device, nvrtc);
}
