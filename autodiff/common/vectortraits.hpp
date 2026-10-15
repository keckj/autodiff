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
#include <autodiff/common/meta.hpp>

// C++ includes
#if !AUTODIFF_NVRTC
#include <vector>
#endif

namespace autodiff {
namespace detail {

/// An auxiliary template type to indicate VectorTraits has not been defined for a type.
template<typename V>
struct VectorTraitsNotDefinedFor {};

/// An auxiliary template type to indicate VectorTraits::ReplaceValueType is not supported for a type.
template<typename V>
struct VectorReplaceValueTypeNotSupportedFor {};

/// A vector traits to be defined for each autodiff number.
template<typename V, class Enable = void>
struct VectorTraits
{
    /// The value type of each entry in the vector.
    using ValueType = VectorTraitsNotDefinedFor<V>;

    /// The template alias to replace the value type of a vector type with another value type.
    using ReplaceValueType = VectorReplaceValueTypeNotSupportedFor<V>;
};

/// A template alias used to get the type of the values in a vector type.
template<typename V>
using VectorValueType = typename VectorTraits<PlainType<V>>::ValueType;

/// A template alias used to get the type of a vector that is equivalent to another but with a different value type.
template<typename V, typename NewValueType>
using VectorReplaceValueType = typename VectorTraits<PlainType<V>>::template ReplaceValueType<NewValueType>;

/// The value used for vector lengths not known at compile time (same as Eigen::Dynamic).
constexpr int DynamicSize = -1;

template<typename V, typename = void>
struct AuxVectorSize { static constexpr int value = DynamicSize; };

template<typename V>
struct AuxVectorSize<V, stdx::void_t<decltype(VectorTraits<V>::Size)>> { static constexpr int value = VectorTraits<V>::Size; };

/// The length of a vector type known at compile time, or DynamicSize (vector traits may define a member `Size`).
template<typename V>
constexpr int VectorSize = AuxVectorSize<PlainType<V>>::value;

/// A compile-time constant that indicates with a type is a vector type.
template<typename V>
constexpr bool isVector = !stdx::is_same_v<VectorValueType<PlainType<V>>, VectorTraitsNotDefinedFor<PlainType<V>>>;


#if !AUTODIFF_NVRTC
/// Implementation of VectorTraits for std::vector.
template<typename T, template<class> typename Allocator>
struct VectorTraits<std::vector<T, Allocator<T>>>
{
    using ValueType = T;

    template<typename NewValueType>
    using ReplaceValueType = std::vector<NewValueType, Allocator<NewValueType>>;
};
#endif

/// Implementation of VectorTraits for stdx::array (std::array or cuda::std::array).
template<typename T, size_t N>
struct VectorTraits<stdx::array<T, N>>
{
    using ValueType = T;

    template<typename NewValueType>
    using ReplaceValueType = stdx::array<NewValueType, N>;

    static constexpr int Size = int(N);
};

template<typename V, typename = void>
struct HasResize : stdx::false_type {};

template<typename V>
struct HasResize<V, stdx::void_t<decltype(stdx::declval<V&>().resize(0))>> : stdx::true_type {};

/// A compile-time constant that indicates if a vector type can be resized (e.g. std::vector, Eigen::VectorXd but not std::array).
template<typename V>
constexpr bool hasResize = HasResize<PlainType<V>>::value;

/// Resize a vector to a given length if its type supports it, otherwise check that its fixed length matches.
template<typename V>
AUTODIFF_DEVICE_FUNC constexpr void resizeVector(V& vec, size_t len)
{
    if constexpr (hasResize<V>)
        vec.resize(len);
    else
        assert(size_t(vec.size()) == len);
}

template<typename M, typename = void>
struct HasResize2 : stdx::false_type {};

template<typename M>
struct HasResize2<M, stdx::void_t<decltype(stdx::declval<M&>().resize(0, 0))>> : stdx::true_type {};

/// Resize a matrix if its type supports it (e.g. Eigen matrices), otherwise check that its fixed dimensions match.
template<typename M>
AUTODIFF_DEVICE_FUNC constexpr void resizeMatrix(M& mat, size_t rows, size_t cols)
{
    if constexpr (HasResize2<PlainType<M>>::value)
        mat.resize(rows, cols);
    else
        assert(size_t(mat.rows()) == rows && size_t(mat.cols()) == cols);
}

} // namespace detail
} // namespace autodiff
