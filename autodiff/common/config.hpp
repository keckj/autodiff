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

//======================================================================================================================
// Compiler detection and portability macros.
//
// AUTODIFF_NVRTC        defined to 1 (0 otherwise) when compiling with NVRTC (CUDA runtime compilation). In this mode there is no
//                       C/C++ standard library: utilities come from libcu++ (<cuda/std/...>) and math functions from
//                       the CUDA math API available in the global namespace.
// AUTODIFF_USE_CUDA_STD defined to 1 when autodiff uses cuda::std instead of std for its utilities (array, tuple,
//                       type traits, ...). Always enabled with NVRTC; can be enabled by the user with nvcc so that host
//                       and device code share the same types (define it to 1 before including any autodiff header).
// AUTODIFF_DEVICE_FUNC  annotation for functions usable on host and device (__host__ __device__ with CUDA/HIP
//                       compilers, empty otherwise). Can be overridden by the user.
// AUTODIFF_HAS_IOSTREAM defined to 1 when <ostream>/<sstream> are available (i.e. not with NVRTC).
// AUTODIFF_VAR_USE_STD_MEMORY
//                       defined to 1 when the reverse mode (autodiff::var) stores its expression trees with
//                       std::shared_ptr and std::function, which are host only. Defaults to 0 with CUDA/HIP compilers
//                       (device capable autodiff::detail::shared_ptr and autodiff::detail::function are used instead,
//                       see autodiff/common/memory.hpp) and to 1 otherwise. Can be overridden by the user (except with
//                       NVRTC).
//
// NVRTC programs including autodiff must be compiled with -default-device (--device-as-default-execution-space):
// autodiff and its users rely on generic lambdas (e.g. [](auto x) { ... }), whose call operators are otherwise
// inferred to be host functions, which NVRTC rejects.
//======================================================================================================================

#if defined(__CUDACC_RTC__)
    #define AUTODIFF_NVRTC 1
#else
    #define AUTODIFF_NVRTC 0
#endif

#if AUTODIFF_NVRTC
    #undef AUTODIFF_USE_CUDA_STD
    #define AUTODIFF_USE_CUDA_STD 1
    #define AUTODIFF_HAS_IOSTREAM 0
#else
    #ifndef AUTODIFF_USE_CUDA_STD
        #define AUTODIFF_USE_CUDA_STD 0
    #endif
    #define AUTODIFF_HAS_IOSTREAM 1
#endif

#if AUTODIFF_NVRTC
    #undef AUTODIFF_VAR_USE_STD_MEMORY
    #define AUTODIFF_VAR_USE_STD_MEMORY 0
#elif !defined(AUTODIFF_VAR_USE_STD_MEMORY)
    #if defined(__CUDACC__) || defined(__HIPCC__)
        #define AUTODIFF_VAR_USE_STD_MEMORY 0
    #else
        #define AUTODIFF_VAR_USE_STD_MEMORY 1
    #endif
#endif

#ifndef AUTODIFF_DEVICE_FUNC
    #if (defined(__CUDACC__) && !defined(EIGEN_NO_CUDA)) || (defined(__HIPCC__) && !defined(EIGEN_NO_HIP))
        #define AUTODIFF_DEVICE_FUNC __host__ __device__
    #else
        #define AUTODIFF_DEVICE_FUNC
    #endif
#endif

// True when the code is currently being compiled for the device (nvcc/NVRTC device pass or HIP device pass)
#if defined(__CUDA_ARCH__) || (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__)
    #define AUTODIFF_DEVICE_COMPILE 1
#else
    #define AUTODIFF_DEVICE_COMPILE 0
#endif

// Report an unrecoverable error: throw on host, assert on device (no exceptions in device code).
#if AUTODIFF_DEVICE_COMPILE || AUTODIFF_NVRTC
    #define AUTODIFF_THROW(exception, msg) assert(false && msg)
#else
    #define AUTODIFF_THROW(exception, msg) throw exception(msg)
#endif

#if AUTODIFF_NVRTC
namespace autodiff::detail {
// Fails to compile with an explicit name when NVRTC is not given -default-device (see the comment at the top).
inline constexpr auto autodiff_with_nvrtc_requires_the_default_device_option = [](auto x) { return x; };
static_assert(autodiff_with_nvrtc_requires_the_default_device_option(true), "compile with NVRTC option -default-device");
} // namespace autodiff::detail
#endif
