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
// The Eigen overloads are enabled if autodiff was configured with Eigen or if Eigen was included before this header
// (as done by autodiff/forward/dual/eigen.hpp and autodiff/forward/real/eigen.hpp).
#if defined(AUTODIFF_EIGEN_FOUND) || defined(EIGEN_WORLD_VERSION)
#include <autodiff/common/eigen.hpp>
#endif
#include <autodiff/common/meta.hpp>
#include <autodiff/common/classtraits.hpp>
#include <autodiff/forward/utils/derivative.hpp>

namespace autodiff {
namespace detail {

/// The length of an item in a `wrt(...)` list known at compile time (DynamicSize if not known).
template<typename Item>
constexpr int WrtItemStaticLength = isVector<Item> ? VectorSize<Item> : 1;

/// The sum of lengths of all items in a `wrt(...)` list known at compile time (DynamicSize if any is not known).
template<typename... Vars>
constexpr int WrtStaticLength = ((WrtItemStaticLength<Vars> == DynamicSize) || ...) ? DynamicSize : (0 + ... + WrtItemStaticLength<Vars>);

/// Return the length of an item in a `wrt(...)` list.
template<typename Item>
AUTODIFF_DEVICE_FUNC auto wrt_item_length(const Item& item) -> size_t
{
    if constexpr (isVector<Item>)
        return item.size(); // if item is a vector, return its size
    else return 1; // if not a vector, say, a number, return 1 for its length
}


/// Return the sum of lengths of all itens in a `wrt(...)` list.
template<typename... Vars>
AUTODIFF_DEVICE_FUNC auto wrt_total_length(const Wrt<Vars...>& wrt) -> size_t
{
    return Reduce(wrt.args, [&](auto&& item) constexpr {
        return wrt_item_length(item);
    });
}

// Loop through each variable in a wrt list and apply a function f(i, x) that
// accepts an index i and the variable x[i], where i is the global index of the
// variable in the list.
template<typename Function, typename... Vars>
AUTODIFF_DEVICE_FUNC constexpr auto ForEachWrtVar(const Wrt<Vars...>& wrt, Function&& f)
{
    auto i = 0; // the current index of the variable in the wrt list
    ForEach(wrt.args, [&](auto& item) constexpr
    {
        using T = decltype(item);
        static_assert(isVector<T> || Order<T> > 0, "Expecting a wrt list with either vectors or individual autodiff numbers.");
        if constexpr (isVector<T>) {
            for(auto j = 0; j < item.size(); ++j)
                // call given f with current index and variable from item (a vector)
                if constexpr (detail::has_operator_bracket<T>()) {
                    f(i++, item[j]);
                } else {
                    f(i++, item(j));
                }
        }
        else f(i++, item); // call given f with current index and variable from item (a number, not a vector)
    });
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args, typename Y, typename G>
AUTODIFF_DEVICE_FUNC void gradient(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, Y& u, G& g)
{
    static_assert(sizeof...(Vars) >= 1);
    static_assert(sizeof...(Args) >= 1);

    const size_t n = wrt_total_length(wrt);

    resizeVector(g, n);

    if(n == 0) return;

    ForEachWrtVar(wrt, [&](auto&& i, auto&& xi) constexpr
    {
        static_assert(!isConst<decltype(xi)>, "Expecting non-const autodiff numbers in wrt list because these need to be seeded, and thus altered!");
        u = eval(f, at, detail::wrt(xi)); // evaluate u with xi seeded so that du/dxi is also computed
        g[i] = derivative<1>(u);
    });

}

#if defined(AUTODIFF_EIGEN_FOUND) || defined(EIGEN_WORLD_VERSION)
/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args, typename Y>
AUTODIFF_DEVICE_FUNC auto gradient(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, Y& u)
{
    using T = NumericType<decltype(u)>; // the underlying numeric floating point type in the autodiff number u
    using Vec = VectorN<T, WrtStaticLength<Vars...>>; // the gradient vector type with floating point values (not autodiff numbers!), with fixed length if known at compile time

    Vec g;
    gradient(f, wrt, at, u, g);
    return g;
}

/// Return the gradient of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args>
AUTODIFF_DEVICE_FUNC auto gradient(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at)
{
    ReturnType<Fun, Args...> u;
    return gradient(f, wrt, at, u);
}
#endif

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Fun, typename... Vars, typename... Args, typename Y, typename Jac>
AUTODIFF_DEVICE_FUNC void jacobian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, Y& F, Jac& J)
{
    static_assert(sizeof...(Vars) >= 1);
    static_assert(sizeof...(Args) >= 1);

    size_t n = wrt_total_length(wrt); /// using const size_t produces an error in GCC 7.3 because of the capture in the constexpr lambda in the ForEach block
    size_t m = 0;

    ForEachWrtVar(wrt, [&](auto&& i, auto&& xi) constexpr {
        static_assert(!isConst<decltype(xi)>, "Expecting non-const autodiff numbers in wrt list because these need to be seeded, and thus altered!");
        F = eval(f, at, detail::wrt(xi)); // evaluate F with xi seeded so that dF/dxi is also computed
        if(m == 0) { m = F.size(); resizeMatrix(J, m, n); };
        for(size_t row = 0; row < m; ++row)
            J(row, i) = derivative<1>(F[row]);
    });
}

#if defined(AUTODIFF_EIGEN_FOUND) || defined(EIGEN_WORLD_VERSION)
/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Fun, typename... Vars, typename... Args, typename Y>
AUTODIFF_DEVICE_FUNC auto jacobian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, Y& F)
{
    using U = VectorValueType<decltype(F)>; // the type of the autodiff numbers in vector F
    using T = NumericType<U>; // the underlying numeric floating point type in the autodiff number U
    using Mat = MatrixMN<T, VectorSize<decltype(F)>, WrtStaticLength<Vars...>>; // the jacobian matrix type with floating point values (not autodiff numbers!), with fixed dimensions if known at compile time

    Mat J;
    jacobian(f, wrt, at, F, J);
    return J;
}

/// Return the Jacobian matrix of a function *f* with respect to some or all variables.
template<typename Fun, typename... Vars, typename... Args>
AUTODIFF_DEVICE_FUNC auto jacobian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at)
{
    using Y = ReturnType<Fun, Args...>;
    static_assert(!stdx::is_same_v<Y, void>,
        "In jacobian(f, wrt(x), at(x)), the type of x "
        "might not be the same as in the definition of f. "
        "For example, x is Eigen::VectorXdual but the "
        "definition of f uses Eigen::Ref<const Eigen::VectorXdual>.");
    Y F;
    return jacobian(f, wrt, at, F);
}
#endif

/// Return the hessian matrix of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args, typename U, typename G, typename H>
AUTODIFF_DEVICE_FUNC void hessian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, U& u, G& g, H& h)
{
    static_assert(sizeof...(Vars) >= 1);
    static_assert(sizeof...(Args) >= 1);

    size_t n = wrt_total_length(wrt);

    resizeVector(g, n);
    resizeMatrix(h, n, n);

    ForEachWrtVar(wrt, [&](auto&& i, auto&& xi) constexpr {
        ForEachWrtVar(wrt, [&](auto&& j, auto&& xj) constexpr
        {
            static_assert(!isConst<decltype(xi)> && !isConst<decltype(xj)>, "Expecting non-const autodiff numbers in wrt list because these need to be seeded, and thus altered!");
            if(j >= i) { // this take advantage of the fact the Hessian matrix is symmetric
                u = eval(f, at, detail::wrt(xi, xj)); // evaluate u with xi and xj seeded to produce u0, du/dxi, d2u/dxidxj
                g[i] = derivative<1>(u);              // get du/dxi from u
                h(i, j) = h(j, i) = derivative<2>(u); // get d2u/dxidxj from u
            }
        });
    });
}

#if defined(AUTODIFF_EIGEN_FOUND) || defined(EIGEN_WORLD_VERSION)
/// Return the hessian matrix of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args, typename U, typename G>
AUTODIFF_DEVICE_FUNC auto hessian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at, U& u, G& g)
{
    using T = NumericType<decltype(u)>; // the underlying numeric floating point type in the autodiff number u
    constexpr auto N = WrtStaticLength<Vars...>;
    using Mat = MatrixMN<T, N, N>; // the Hessian matrix type with floating point values (not autodiff numbers!), with fixed dimensions if known at compile time

    Mat H;
    hessian(f, wrt, at, u, g, H);
    return H;
}

/// Return the hessian matrix of scalar function *f* with respect to some or all variables *x*.
template<typename Fun, typename... Vars, typename... Args>
AUTODIFF_DEVICE_FUNC auto hessian(const Fun& f, const Wrt<Vars...>& wrt, const At<Args...>& at)
{
    using U = ReturnType<Fun, Args...>;
    using T = NumericType<U>;
    using Vec = VectorN<T, WrtStaticLength<Vars...>>;
    U u;
    Vec g;
    return hessian(f, wrt, at, u, g);
}
#endif

} // namespace detail

using detail::gradient;
using detail::jacobian;
using detail::hessian;

} // namespace autodiff

