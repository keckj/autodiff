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
#include <autodiff/common/stdlib.hpp>

namespace autodiff {
namespace detail {

/// The maximum order for binomial coefficients supported currently (50).
constexpr size_t binomialcoeffs_nmax = 50;

/// Return the binomial coefficient C(i,j), exactly computed in double precision for i <= binomialcoeffs_nmax.
/// @note Computed instead of being read from a table, because NVRTC cannot read namespace scope arrays in device constant expressions.
AUTODIFF_DEVICE_FUNC constexpr double binomialCoefficient(size_t i, size_t j)
{
    if(j > i) return 0.0;
    if(j > i - j) j = i - j;
    double res = 1.0;
    for(size_t k = 1; k <= j; ++k)
        res = res * double(i - j + k) / double(k); // res = C(i-j+k, k), all intermediate values are exact integers
    return res;
}

/// Return the binomial coefficient C(i,j) at compile time.
template<size_t i, size_t j>
struct AuxBinomialCoefficient
{
    static_assert(i <= binomialcoeffs_nmax, "Violation of maximum order for binomial coefficient retrieval.");
    static_assert(j <= i, "Violation of j <= i condition for retrieving binomial coefficient C(i,j).");
    constexpr static double value = binomialCoefficient(i, j);
};

/// The binomial coefficient C(i,j) at compile time.
template<size_t i, size_t j>
constexpr double BinomialCoefficient = AuxBinomialCoefficient<i, j>::value;

} // namespace detail
} // namespace autodiff
