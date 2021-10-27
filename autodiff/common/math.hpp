//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2020 Allan Leal
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

#ifndef MATHS_HPP_U7IYJWBZ
#define MATHS_HPP_U7IYJWBZ

namespace autodiff {
namespace detail {

template<typename T>
__constant__ constexpr T pi(3.141592653589793238462643383279502884197169399375105820974944592307816406286);

template <typename T>
__constant__ constexpr T ln10(2.3025850929940456840179914546843);
    
template <typename T>
__constant__ constexpr T sqrt_pi(1.7724538509055160272981674833411451872554456638435);

namespace math {

template <typename T>
__forceinline__ __host__ __device__ T hypot(T x, T y, T z) {
#ifdef __CUDA_ARCH__
    return ::norm3d(x,y,z);
#else
    return ::hypot(x,y,z);
#endif
}

} /* end of namespace math */
} /* end of namespace detail */
} /* end of namespace autodiff */

#endif /* end of include guard: MATHS_HPP_U7IYJWBZ */
