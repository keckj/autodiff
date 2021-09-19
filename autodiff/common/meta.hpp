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

#ifndef META_HPP_PQ3JSZOE
#define META_HPP_PQ3JSZOE

#include <cuda/std/type_traits>
#include <cuda/std/cstddef>
#include <cuda/std/tuple>

namespace autodiff {
namespace detail {

template<bool value, typename T=bool>
using EnableIf = typename cuda::std::enable_if<value, T>::type;

template<typename T>
using PlainType = typename cuda::std::remove_cv<typename cuda::std::remove_reference<T>::type>::type;

template<bool Cond, typename WhenTrue, typename WhenFalse>
using ConditionalType = typename cuda::std::conditional<Cond, WhenTrue, WhenFalse>::type;

template<typename A, typename B>
using CommonType = typename cuda::std::common_type<A, B>::type;

template<typename Fun, typename... Args>
using ReturnType = cuda::std::invoke_result_t<Fun, Args...>;

template<typename T, typename U>
constexpr bool isConvertible = cuda::std::is_convertible<PlainType<T>, U>::value;

template<typename A, typename B>
constexpr bool isSame = cuda::std::is_same_v<A, B>;

template<typename Tuple>
constexpr auto TupleSize = cuda::std::tuple_size_v<cuda::std::decay_t<Tuple>>;

template<typename Tuple>
__host__ __device__ constexpr auto TupleHead(Tuple&& tuple)
{
    return cuda::std::get<0>(cuda::std::forward<Tuple>(tuple));
}

template<typename Tuple>
__host__ __device__ constexpr auto TupleTail(Tuple&& tuple)
{
    auto g = [&](auto&&, auto&&... args) constexpr {
        return cuda::std::forward_as_tuple(args...);
    };
    return cuda::std::apply(g, cuda::std::forward<Tuple>(tuple));
}

template<size_t i>
struct Index
{
    constexpr static size_t index = i;
    __host__ __device__ constexpr operator size_t() const { return index; }
    __host__ __device__ constexpr operator size_t() { return index; }
};

template<size_t i, size_t ibegin, size_t iend, typename Function>
__host__ __device__ constexpr void AuxFor(Function&& f)
{
    if constexpr (i < iend) {
        f(Index<i>{});
        AuxFor<i + 1, ibegin, iend>(cuda::std::forward<Function>(f));
    }
}

template<size_t ibegin, size_t iend, typename Function>
__host__ __device__ constexpr void For(Function&& f)
{
    AuxFor<ibegin, ibegin, iend>(cuda::std::forward<Function>(f));
}

template<size_t iend, typename Function>
__host__ __device__ constexpr void For(Function&& f)
{
    For<0, iend>(cuda::std::forward<Function>(f));
}

template<size_t i, size_t ibegin, size_t iend, typename Function>
__host__ __device__ constexpr void AuxReverseFor(Function&& f)
{
    if constexpr (i < iend)
    {
        AuxReverseFor<i + 1, ibegin, iend>(cuda::std::forward<Function>(f));
        f(Index<i>{});
    }
}

template<size_t ibegin, size_t iend, typename Function>
__host__ __device__ constexpr void ReverseFor(Function&& f)
{
    AuxReverseFor<ibegin, ibegin, iend>(cuda::std::forward<Function>(f));
}

template<size_t iend, typename Function>
__host__ __device__ constexpr void ReverseFor(Function&& f)
{
    ReverseFor<0, iend>(cuda::std::forward<Function>(f));
}

template<typename Tuple, typename Function>
__host__ __device__ constexpr void ForEach(Tuple&& tuple, Function&& f)
{
    constexpr auto N = TupleSize<Tuple>;
    For<N>([&](auto i) constexpr {
        f(cuda::std::get<i>(tuple));
    });
    //------------------------------------------------------------
    // ALTERNATIVE IMPLEMENTATION POSSIBLY USEFUL TO KEEP IT HERE
    // auto g = [&](auto&&... args) constexpr {
    //     ( f(cuda::std::forward<decltype(args)>(args)), ...);
    // };
    // cuda::std::apply(g, cuda::std::forward<Tuple>(tuple));
    //------------------------------------------------------------
}

template<typename Tuple1, typename Tuple2, typename Function>
__host__ __device__ constexpr auto ForEach(Tuple1&& tuple1, Tuple2&& tuple2, Function&& f)
{
    constexpr auto N1 = TupleSize<Tuple1>;
    constexpr auto N2 = TupleSize<Tuple2>;
    static_assert(N1 == N2);
    For<N1>([&](auto i) constexpr {
        f(cuda::std::get<i>(tuple1), cuda::std::get<i>(tuple2));
    });
}

template<size_t ibegin, size_t iend, typename Function>
__host__ __device__ constexpr auto Sum(Function&& f)
{
    using ResultType = cuda::std::invoke_result_t<Function, Index<ibegin>>;
    ResultType res = {};
    For<ibegin, iend>([&](auto i) constexpr {
        res += f(Index<i>{});
    });
    return res;
}

template<typename Tuple, typename Function>
__host__ __device__ constexpr auto Reduce(Tuple&& tuple, Function&& f)
{
    using ResultType = cuda::std::invoke_result_t<Function, decltype(cuda::std::get<0>(tuple))>;
    ResultType res = {};
    ForEach(tuple, [&](auto&& item) constexpr {
        res += f(item);
    });
    return res;
}

} // namespace detail
} // namespace autodiff

#endif /* end of include guard: META_HPP_PQ3JSZOE */

