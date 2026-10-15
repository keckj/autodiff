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

#pragma once

// autodiff includes
#include <autodiff/common/config.hpp>

//======================================================================================================================
// Standard library abstraction layer.
//
// autodiff code uses `stdx::` (alias of `std` or `cuda::std`) for library utilities and `stdmath::` for math functions,
// so that the same headers compile with a host compiler, with nvcc/hipcc and with NVRTC.
//======================================================================================================================

#if AUTODIFF_USE_CUDA_STD
    #include <cuda/std/array>
    #include <cuda/std/cassert>
    #include <cuda/std/cstddef>
    #include <cuda/std/functional>
    #include <cuda/std/tuple>
    #include <cuda/std/type_traits>
    #include <cuda/std/utility>
#else
    #include <array>
    #include <cassert>
    #include <cstddef>
    #include <functional>
    #include <tuple>
    #include <type_traits>
    #include <utility>
#endif

#if !AUTODIFF_NVRTC
    #include <algorithm>
    #include <cmath>
#endif

namespace autodiff {

#if AUTODIFF_USE_CUDA_STD
namespace stdx = ::cuda::std;
#else
namespace stdx = ::std;
#endif

namespace stdmath {

#if AUTODIFF_NVRTC
// NVRTC: the CUDA math API is declared in the global namespace (float and double overloads)
using ::abs;
using ::acos;
using ::acosh;
using ::asin;
using ::asinh;
using ::atan;
using ::atan2;
using ::atanh;
using ::cbrt;
using ::ceil;
using ::copysign;
using ::cos;
using ::cosh;
using ::erf;
using ::erfc;
using ::exp;
using ::exp2;
using ::expm1;
using ::floor;
using ::fmax;
using ::fmin;
using ::fmod;
using ::hypot;
using ::log;
using ::log10;
using ::log1p;
using ::log2;
using ::max;
using ::min;
using ::pow;
using ::round;
using ::sin;
using ::sinh;
using ::sqrt;
using ::tan;
using ::tanh;
using ::trunc;
#else
using std::abs;
using std::acos;
using std::acosh;
using std::asin;
using std::asinh;
using std::atan;
using std::atan2;
using std::atanh;
using std::cbrt;
using std::ceil;
using std::copysign;
using std::cos;
using std::cosh;
using std::erf;
using std::erfc;
using std::exp;
using std::exp2;
using std::expm1;
using std::floor;
using std::fmax;
using std::fmin;
using std::fmod;
using std::hypot;
using std::log;
using std::log10;
using std::log1p;
using std::log2;
using std::max;
using std::min;
using std::pow;
using std::round;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;
using std::trunc;
#endif

/// Return the hypotenuse sqrt(x*x + y*y + z*z) (the three arguments overload of std::hypot is not available on device).
template<typename T>
AUTODIFF_DEVICE_FUNC auto hypot3(const T& x, const T& y, const T& z)
{
#if AUTODIFF_DEVICE_COMPILE || AUTODIFF_NVRTC
    return ::norm3d(x, y, z);
#else
    return std::hypot(x, y, z);
#endif
}

} // namespace stdmath
} // namespace autodiff
