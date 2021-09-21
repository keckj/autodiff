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

#ifndef EIGEN_HPP_KU8GYPEF
#define EIGEN_HPP_KU8GYPEF

// Eigen includes
#include <Eigen/Core>

// autodiff includes
#include <autodiff/common/vectortraits.hpp>

//=====================================================================================================================
//
// EIGEN MACROS FOR CREATING NEW TYPE ALIASES
//
//=====================================================================================================================

#define AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)                        \
using Array##SizeSuffix##SizeSuffix##TypeSuffix = Eigen::Array<Type, Size, Size>;                 \
using Array##SizeSuffix##TypeSuffix             = Eigen::Array<Type, Size, 1>;                    \
using Matrix##SizeSuffix##TypeSuffix            = Eigen::Matrix<Type, Size, Size, 0, Size, Size>; \
using Vector##SizeSuffix##TypeSuffix            = Eigen::Matrix<Type, Size, 1, 0, Size, 1>;       \
using RowVector##SizeSuffix##TypeSuffix         = Eigen::Matrix<Type, 1, Size, 1, 1, Size>;

#define AUTODIFF_DEFINE_EIGEN_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 2, 2)             \
AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 3, 3)             \
AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 4, 4)             \
AUTODIFF_DEFINE_EIGEN_TYPEDEFS(Type, TypeSuffix, 5, 5)

namespace autodiff {
namespace detail {

//=====================================================================================================================
//
// DEFINE VECTOR TRAITS FOR EIGEN TYPES
//
//=====================================================================================================================

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct VectorTraits<Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
{
    static constexpr size_t size = Rows*Cols;

    using ValueType = Scalar;

    template<typename NewValueType>
    using ReplaceValueType = Eigen::Matrix<NewValueType, Rows, Cols, Options, MaxRows, MaxCols>;
};

template<typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
struct VectorTraits<Eigen::Array<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>
{
    static constexpr size_t size = Rows*Cols;

    using ValueType = Scalar;

    template<typename NewValueType>
    using ReplaceValueType = Eigen::Array<NewValueType, Rows, Cols, Options, MaxRows, MaxCols>;
};

template<typename VectorType, int Size>
struct VectorTraits<Eigen::VectorBlock<VectorType, Size>>
{
    static_assert(Size >= 0, "Cannot use a dynamic vector block as a fixed size vector.");
    static constexpr size_t size = Size;

    using ValueType = typename PlainType<VectorType>::Scalar;

    template<typename NewValueType>
    using ReplaceValueType = VectorReplaceValueType<VectorType, NewValueType>;
};

#if 0
template<typename VectorType, typename IndicesType>
struct VectorTraits<Eigen::IndexedView<VectorType, IndicesType, Eigen::internal::SingleRange>>
{
    static constexpr size_t size = typename PlainType<VectorType>::size;

    using ValueType = typename PlainType<VectorType>::Scalar;

    template<typename NewValueType>
    using ReplaceValueType = VectorReplaceValueType<VectorType, NewValueType>;
};

template<typename VectorType, typename IndicesType>
struct VectorTraits<Eigen::IndexedView<VectorType, Eigen::internal::SingleRange, IndicesType>>
{
    static constexpr size_t size = typename PlainType<VectorType>::size;

    using ValueType = typename PlainType<VectorType>::Scalar;

    template<typename NewValueType>
    using ReplaceValueType = VectorReplaceValueType<VectorType, NewValueType>;
};
#endif

template<typename MatrixType>
struct VectorTraits<Eigen::Ref<MatrixType>>
{
    static constexpr size_t size = VectorTraits<PlainType<MatrixType>>::size;

    using ValueType = VectorValueType<MatrixType>;

    template<typename NewValueType>
    using ReplaceValueType = VectorReplaceValueType<MatrixType, NewValueType>;
};

//=====================================================================================================================
//
// AUXILIARY TEMPLATE TYPE ALIASES
//
//=====================================================================================================================

template<typename Scalar, int N>
using Vector = Eigen::Matrix<Scalar, N, 1, 0, N, 1>;

template<typename Scalar, int M, int N=M>
using Matrix = Eigen::Matrix<Scalar, M, N, 0, M, N>;

template<typename Scalar, int M, int N=1>
using Array = Eigen::Array<Scalar, M, N, 0, M, N>;


} // namespace detail

using detail::Vector;
using detail::Array;
using detail::Matrix;

} // namespace autodiff

#endif /* end of include guard: EIGEN_HPP_KU8GYPEF */
