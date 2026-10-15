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
// of this software and `associated documentation files (the "Software"), to deal
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

// C++ includes
#include <autodiff/common/stdlib.hpp>
#if AUTODIFF_HAS_IOSTREAM
#include <iostream>
#include <sstream>
#endif

// autodiff includes
#include <autodiff/common/binomialcoefficient.hpp>
#include <autodiff/common/numbertraits.hpp>
#include <autodiff/common/meta.hpp>

namespace autodiff {
namespace detail {

/// The type used to represent a real number that supports up to *N*-th order derivative calculation.
template<size_t N, typename T>
class Real
{
private:
    // Ensure type T is a numeric type
    static_assert(isArithmetic<T>);

    /// The value and derivatives of the number up to order *N*.
    stdx::array<T, N + 1> m_data = {};

public:
    /// Construct a default Real number of order *N* and type *T*.
    AUTODIFF_DEVICE_FUNC constexpr Real()
    {}

    /// Construct a Real number with given data.
    AUTODIFF_DEVICE_FUNC constexpr Real(const T& value)
    {
        m_data[0] = value;
    }

    /// Construct a Real number with given data.
    AUTODIFF_DEVICE_FUNC constexpr Real(const stdx::array<T, N + 1>& data)
    : m_data(data)
    {}

    /// Construct a Real number with given data.
    template<size_t M, typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr explicit Real(const Real<M, U>& other)
    {
        static_assert(N <= M);
        For<0, N + 1>([&](auto i) constexpr {
            m_data[i] = static_cast<T>(other[i]);
        });
    }

    /// Return the value of the Real number.
    AUTODIFF_DEVICE_FUNC constexpr auto val() -> T&
    {
        return m_data[0];
    }

    /// Return the value of the Real number.
    AUTODIFF_DEVICE_FUNC constexpr auto val() const -> const T&
    {
        return m_data[0];
    }

    AUTODIFF_DEVICE_FUNC constexpr auto operator[](size_t i) -> T&
    {
        return m_data[i];
    }

    AUTODIFF_DEVICE_FUNC constexpr auto operator[](size_t i) const -> const T&
    {
        return m_data[i];
    }

    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr auto operator=(const U& value) -> Real&
    {
        m_data[0] = value;
        For<1, N + 1>([&](auto i) constexpr { m_data[i] = T{}; });
        return *this;
    }

    AUTODIFF_DEVICE_FUNC constexpr auto operator=(const stdx::array<T, N + 1>& data)
    {
        m_data = data;
        return *this;
    }

    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr auto operator+=(const U& value) -> Real&
    {
        m_data[0] += static_cast<T>(value);
        return *this;
    }

    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr auto operator-=(const U& value) -> Real&
    {
        m_data[0] -= static_cast<T>(value);
        return *this;
    }

    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr auto operator*=(const U& value) -> Real&
    {
        For<0, N + 1>([&](auto i) constexpr { m_data[i] *= static_cast<T>(value); });
        return *this;
    }

    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr auto operator/=(const U& value) -> Real&
    {
        For<0, N + 1>([&](auto i) constexpr { m_data[i] /= static_cast<T>(value); });
        return *this;
    }

    AUTODIFF_DEVICE_FUNC constexpr auto operator+=(const Real& y)
    {
        auto& x = *this;
        For<0, N + 1>([&](auto i) constexpr { x[i] += y[i]; });
        return *this;
    }

    AUTODIFF_DEVICE_FUNC constexpr auto operator-=(const Real& y)
    {
        auto& x = *this;
        For<0, N + 1>([&](auto i) constexpr { x[i] -= y[i]; });
        return *this;
    }

    AUTODIFF_DEVICE_FUNC constexpr auto operator*=(const Real& y)
    {
        auto& x = *this;
        ReverseFor<N + 1>([&](auto i) constexpr {
            x[i] = Sum<0, i + 1>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index, decltype(j)::index>;
                return c * x[i - j] * y[j];
            });
        });
        return *this;
    }

    AUTODIFF_DEVICE_FUNC constexpr auto operator/=(const Real& y)
    {
        auto& x = *this;
        For<N + 1>([&](auto i) constexpr {
            x[i] -= Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index, decltype(j)::index>;
                return c * x[j] * y[i - j];
            });
            x[i] /= y[0];
        });
        return *this;
    }

#if defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION_REAL) || defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION)
    AUTODIFF_DEVICE_FUNC constexpr operator T() const { return static_cast<T>(m_data[0]); }

    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr operator U() const { return static_cast<U>(m_data[0]); }
#else
    AUTODIFF_DEVICE_FUNC constexpr explicit operator T() const { return static_cast<T>(m_data[0]); }

    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC constexpr explicit operator U() const { return static_cast<U>(m_data[0]); }
#endif
};

//=====================================================================================================================
//
// STANDARD TEMPLATE LIBRARY MATH FUNCTIONS
//
//=====================================================================================================================

using stdmath::abs;
using stdmath::acos;
using stdmath::acosh;
using stdmath::asin;
using stdmath::asinh;
using stdmath::atan2;
using stdmath::copysign;
using stdmath::atan;
using stdmath::atanh;
using stdmath::cbrt;
using stdmath::cos;
using stdmath::cosh;
using stdmath::exp;
using stdmath::log;
using stdmath::log10;
using stdmath::max;
using stdmath::min;
using stdmath::pow;
using stdmath::sin;
using stdmath::sinh;
using stdmath::sqrt;
using stdmath::tan;
using stdmath::tanh;
using stdmath::ceil;
using stdmath::erf;
using stdmath::erfc;
using stdmath::exp2;
using stdmath::expm1;
using stdmath::floor;
using stdmath::fmax;
using stdmath::fmin;
using stdmath::fmod;
using stdmath::hypot;
using stdmath::log1p;
using stdmath::log2;
using stdmath::round;
using stdmath::trunc;

//=====================================================================================================================
//
// TYPE TRAITS
//
//=====================================================================================================================

template<typename T>
struct isRealAux { constexpr static bool value = false; };

template<size_t N, typename T>
struct isRealAux<Real<N, T>> { constexpr static bool value = true; };

template<typename T>
constexpr bool isReal = isRealAux<PlainType<T>>::value;

template<typename... Args>
constexpr bool areReal = (... && isReal<Args>);

/// True if at least one argument is a real number and all others are real or arithmetic numbers.
template<typename... Args>
constexpr bool isRealOperable = (... || isReal<Args>) && (... && (isReal<Args> || isArithmetic<Args>));

/// Return the value of a real number or an arithmetic number (used by the functions with mixed arguments).
template<typename X>
AUTODIFF_DEVICE_FUNC constexpr auto realValue(const X& x)
{
    if constexpr (isReal<X>)
        return x[0];
    else return x;
}

//=====================================================================================================================
//
// UNARY OPERATORS +(Real) AND -(Real)
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto operator+(const Real<N, T>& x)
{
    return x;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto operator-(const Real<N, T>& x)
{
    Real<N, T> res;
    For<0, N + 1>([&](auto i) constexpr { res[i] = -x[i]; });
    return res;
}

//=====================================================================================================================
//
// BINARY OPERATOR +(Real, Real), +(Real, Number), +(Number, Real)
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto operator+(Real<N, T> x, const Real<N, T>& y)
{
    return x += y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC auto operator+(Real<N, T> x, const U& y)
{
    return x += y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC auto operator+(const U& x, Real<N, T> y)
{
    return y += x;
}

//=====================================================================================================================
//
// BINARY OPERATOR -(Real, Real), -(Real, Number), -(Number, Real)
//
//=====================================================================================================================
template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto operator-(Real<N, T> x, const Real<N, T>& y)
{
    return x -= y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC auto operator-(Real<N, T> x, const U& y)
{
    return x -= y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC auto operator-(const U& x, Real<N, T> y)
{
    y -= x;
    y *= -static_cast<T>(1.0);
    return y;
}

//=====================================================================================================================
//
// BINARY OPERATOR *(Real, Real), *(Real, Number), *(Number, Real)
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto operator*(Real<N, T> x, const Real<N, T>& y)
{
    return x *= y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC auto operator*(Real<N, T> x, const U& y)
{
    return x *= y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC auto operator*(const U& x, Real<N, T> y)
{
    return y *= x;
}

//=====================================================================================================================
//
// BINARY OPERATOR /(Real, Real), /(Real, Number), /(Number, Real)
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto operator/(Real<N, T> x, const Real<N, T>& y)
{
    return x /= y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC auto operator/(Real<N, T> x, const U& y)
{
    return x /= y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC auto operator/(const U& x, Real<N, T> y)
{
    Real<N, T> z = x;
    return z /= y;
}

//=====================================================================================================================
//
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto exp(const Real<N, T>& x)
{
    Real<N, T> expx;
    expx[0] = exp(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        expx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
            return c * x[i - j] * expx[j];
        });
    });
    return expx;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto log(const Real<N, T>& x)
{
    assert(x[0] != 0 && "autodiff::log(x) has undefined value and derivatives when x = 0");
    Real<N, T> logx;
    logx[0] = log(x[0]);
    For<1, N + 1>([&](auto i) constexpr {
        logx[i] = x[i] - Sum<1, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index - 1>;
            return c * x[i - j] * logx[j];
        });
        logx[i] /= x[0];
    });
    return logx;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto log10(const Real<N, T>& x)
{
    assert(x[0] != 0 && "autodiff::log10(x) has undefined value and derivatives when x = 0");
    const auto ln10 = 2.302585092994046;
    Real<N, T> res = log(x);
    return res /= ln10;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto log2(const Real<N, T>& x)
{
    assert(x[0] != 0 && "autodiff::log2(x) has undefined value and derivatives when x = 0");
    const auto ln2 = 0.69314718055994530941723212145818;
    Real<N, T> res = log(x);
    res /= ln2;
    res[0] = log2(x[0]);
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto log1p(const Real<N, T>& x)
{
    Real<N, T> res = log(1 + x); // same derivatives, log1p(x[0]) is more accurate when x[0] is small
    res[0] = log1p(x[0]);
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto exp2(const Real<N, T>& x)
{
    const auto ln2 = 0.69314718055994530941723212145818;
    Real<N, T> res = exp(ln2 * x);
    res[0] = exp2(x[0]);
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto expm1(const Real<N, T>& x)
{
    Real<N, T> res = exp(x); // same derivatives, expm1(x[0]) is more accurate when x[0] is small
    res[0] = expm1(x[0]);
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto sqrt(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = sqrt(x[0]);

    if constexpr (N > 0)
    {
        // assert(x[0] != 0 && "autodiff::sqrt(x) has undefined derivatives when x = 0");
        if(x[0] == 0) return res;
        Real<N, T> a;
        For<1, N + 1>([&](auto i) constexpr {
            a[i] = x[i] - Sum<1, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index - 1>;
                return c * x[i - j] * a[j];
            });
            a[i] /= x[0];

            res[i] = 0.5 * Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * a[i - j] * res[j];
            });
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto cbrt(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = cbrt(x[0]);

    if constexpr (N > 0)
    {
        // assert(x[0] != 0 && "autodiff::cbrt(x) has undefined derivatives when x = 0");
        if(x[0] == 0) return res;
        Real<N, T> a;
        For<1, N + 1>([&](auto i) constexpr {
            a[i] = x[i] - Sum<1, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index - 1>;
                return c * x[i - j] * a[j];
            });
            a[i] /= x[0];

            res[i] = (1.0/3.0) * Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * a[i - j] * res[j];
            });
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto pow(const Real<N, T>& x, const Real<N, T>& y)
{
    Real<N, T> res;
    res[0] = pow(x[0], y[0]);
    if constexpr (N > 0)
    {
        // assert(x[0] != 0 && "autodiff::pow(x, y) has undefined derivatives when x = 0");
        if(x[0] == 0) return res;
        Real<N, T> lnx = log(x);
        Real<N, T> a;
        For<1, N + 1>([&](auto i) constexpr {
            a[i] = Sum<0, i + 1>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index, decltype(j)::index>;
                return c * y[i - j] * lnx[j];
            });

            res[i] = Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * a[i - j] * res[j];
            });
        });
    }
    return res;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto pow(const Real<N, T>& x, const U& c)
{
    Real<N, T> res;
    res[0] = pow(x[0], static_cast<T>(c));
    if constexpr (N > 0)
    {
        // assert(x[0] != 0 && "autodiff::pow(x, y) has undefined derivatives when x = 0");
        if(x[0] == 0) return res;
        Real<N, T> a = c * log(x);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * a[i - j] * res[j];
            });
        });
    }
    return res;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto pow(const U& c, const Real<N, T>& y)
{
    Real<N, T> res;
    res[0] = pow(static_cast<T>(c), y[0]);
    if constexpr (N > 0)
    {
        // assert(c != 0 && "autodiff::pow(x, y) has undefined derivatives when x = 0");
        if(c == 0) return res;
        Real<N, T> a = y * log(c);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * a[i - j] * res[j];
            });
        });
    }
    return res;
}

//=====================================================================================================================
//
// TRIGONOMETRIC FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto sincos(const Real<N, T>& x) -> stdx::tuple<Real<N, T>, Real<N, T>>
{
    Real<N, T> sinx;
    Real<N, T> cosx;

    cosx[0] = cos(x[0]);
    sinx[0] = sin(x[0]);

    For<1, N + 1>([&](auto i) constexpr {
        cosx[i] = -Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
            return c * x[i - j] * sinx[j];
        });

        sinx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
            return c * x[i - j] * cosx[j];
        });
    });

    return {sinx, cosx};
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto sin(const Real<N, T>& x)
{
    return stdx::get<0>(sincos(x));
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto cos(const Real<N, T>& x)
{
    return stdx::get<1>(sincos(x));
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto tan(const Real<N, T>& x)
{
    Real<N, T> tanx;
    tanx[0] = tan(x[0]);

    if constexpr (N > 0)
    {
        Real<N, T> aux;
        aux[0] = 1 + tanx[0]*tanx[0];

        For<1, N + 1>([&](auto i) constexpr {
            tanx[i] = Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * x[i - j] * aux[j];
            });

            aux[i] = 2*Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * tanx[i - j] * tanx[j];
            });
        });
    }

    return tanx;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto asin(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = asin(x[0]);
    if constexpr (N > 0)
    {
        assert(x[0] < 1.0 && "autodiff::asin(x) has undefined derivative when |x| >= 1");
        Real<N - 1, T> xprime;
        For<1, N + 1>([&](auto i) constexpr {
            xprime[i - 1] = x[i];
        });
        Real<N - 1, T> aux(x);
        aux = xprime/sqrt(1 - aux*aux);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto acos(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = acos(x[0]);
    if constexpr (N > 0)
    {
        assert(x[0] < 1.0 && "autodiff::acos(x) has undefined derivative when |x| >= 1");
        Real<N - 1, T> xprime;
        For<1, N + 1>([&](auto i) constexpr {
            xprime[i - 1] = x[i];
        });
        Real<N - 1, T> aux(x);
        aux = -xprime/sqrt(1 - aux*aux);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto atan(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = atan(x[0]);
    if constexpr (N > 0)
    {
        Real<N - 1, T> xprime;
        For<1, N + 1>([&](auto i) constexpr {
            xprime[i - 1] = x[i];
        });
        Real<N - 1, T> aux(x);
        aux = xprime/(1 + aux*aux);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto atan2(const U& c, const Real<N, T>& x)
{
    // d[atan2(c,x)]/dx = -c / (c^2 + x^2)
    Real<N, T> res;
    res[0] = atan2(c, x[0]);
    if constexpr(N > 0) {
        Real<N - 1, T> xprime;
        For<1, N + 1>([&](auto i) constexpr {
            xprime[i - 1] = x[i];
        });
        Real<N - 1, T> aux(x);
        aux = xprime * (-c / (c * c + aux * aux));
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto atan2(const Real<N, T>& y, const U& c)
{
    // d[atan2(y,c)]/dy = c / (c^2 + y^2)
    Real<N, T> res;
    res[0] = atan2(y[0], c);
    if constexpr(N > 0) {
        Real<N - 1, T> yprime;
        For<1, N + 1>([&](auto i) constexpr {
            yprime[i - 1] = y[i];
        });
        Real<N - 1, T> aux(y);
        aux = yprime * (c / (c * c + aux * aux));
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto atan2(const Real<N, T>& y, const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = atan2(y[0], x[0]);
    if constexpr(N > 0) {
        const T denom = x[0] * x[0] + y[0] * y[0];
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = (x[0] * y[i] - x[i] * y[0]) / denom;
        });
    }
    return res;
}

//=====================================================================================================================
//
// HYPERBOLIC FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto sinhcosh(const Real<N, T>& x) -> stdx::tuple<Real<N, T>, Real<N, T>>
{
    Real<N, T> sinhx;
    Real<N, T> coshx;

    coshx[0] = cosh(x[0]);
    sinhx[0] = sinh(x[0]);

    For<1, N + 1>([&](auto i) constexpr {
        coshx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
            return c * x[i - j] * sinhx[j];
        });

        sinhx[i] = Sum<0, i>([&](auto j) constexpr {
            constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
            return c * x[i - j] * coshx[j];
        });
    });

    return {sinhx, coshx};
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto sinh(const Real<N, T>& x)
{
    return stdx::get<0>(sinhcosh(x));
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto cosh(const Real<N, T>& x)
{
    return stdx::get<1>(sinhcosh(x));
}


template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC auto tanh(const Real<N, T>& x)
{
    Real<N, T> tanhx;
    tanhx[0] = tanh(x[0]);

    if constexpr (N > 0)
    {
        Real<N, T> aux;

        aux[0] = 1 - tanhx[0]*tanhx[0];

        For<1, N + 1>([&](auto i) constexpr {
            tanhx[i] = Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * x[i - j] * aux[j];
            });

            aux[i] = -2*Sum<0, i>([&](auto j) constexpr {
                constexpr auto c = BinomialCoefficient<decltype(i)::index - 1, decltype(j)::index>;
                return c * tanhx[i - j] * tanhx[j];
            });
        });
    }

    return tanhx;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto asinh(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = asinh(x[0]);
    if constexpr (N > 0)
    {
        Real<N - 1, T> xprime;
        For<1, N + 1>([&](auto i) constexpr {
            xprime[i - 1] = x[i];
        });
        Real<N - 1, T> aux(x);
        aux = xprime/sqrt(aux*aux + 1);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto acosh(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = acosh(x[0]);
    if constexpr (N > 0)
    {
        assert(x[0] > 1.0 && "autodiff::acosh(x) has undefined derivative when |x| <= 1");
        Real<N - 1, T> xprime;
        For<1, N + 1>([&](auto i) constexpr {
            xprime[i - 1] = x[i];
        });
        Real<N - 1, T> aux(x);
        aux = xprime/sqrt(aux*aux - 1);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto atanh(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = atanh(x[0]);
    if constexpr (N > 0)
    {
        assert(x[0] < 1.0 && "autodiff::atanh(x) has undefined derivative when |x| >= 1");
        Real<N - 1, T> xprime;
        For<1, N + 1>([&](auto i) constexpr {
            xprime[i - 1] = x[i];
        });
        Real<N - 1, T> aux(x);
        aux = xprime/(1 - aux*aux);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

//=====================================================================================================================
//
// OTHER FUNCTIONS
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto abs(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = abs(x[0]);
    if constexpr (N > 0)
    {
        // assert(x[0] != 0 && "autodiff::abs(x) has undefined derivatives when x = 0");
        if(x[0] == 0) return res;
        const T s = copysign(T(1), x[0]);
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = s * x[i];
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto abs2(const Real<N, T>& x)
{
    return x * x;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto conj(const Real<N, T>& x)
{
    return x;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto erf(const Real<N, T>& x)
{
    Real<N, T> res;
    res[0] = erf(x[0]);
    if constexpr (N > 0)
    {
        const auto two_over_sqrt_pi = 1.1283791670955125738961589031215451716881012586580;
        Real<N - 1, T> xprime;
        For<1, N + 1>([&](auto i) constexpr {
            xprime[i - 1] = x[i];
        });
        Real<N - 1, T> aux(x);
        aux = two_over_sqrt_pi * exp(-aux*aux) * xprime;
        For<1, N + 1>([&](auto i) constexpr {
            res[i] = aux[i - 1];
        });
    }
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto erfc(const Real<N, T>& x)
{
    Real<N, T> res = -erf(x);
    res[0] = erfc(x[0]);
    return res;
}

template<typename X, typename Y, Requires<isRealOperable<X, Y>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto hypot(const X& x, const Y& y)
{
    auto res = sqrt(x * x + y * y);
    res[0] = hypot(realValue(x), realValue(y));
    return res;
}

template<typename X, typename Y, typename Z, Requires<isRealOperable<X, Y, Z>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto hypot(const X& x, const Y& y, const Z& z)
{
    auto res = sqrt(x * x + y * y + z * z);
    using T = PlainType<decltype(res[0])>;
    res[0] = stdmath::hypot3(static_cast<T>(realValue(x)), static_cast<T>(realValue(y)), static_cast<T>(realValue(z)));
    return res;
}

template<typename X, typename Y, Requires<isRealOperable<X, Y>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto fmod(const X& x, const Y& y)
{
    const auto rem = fmod(realValue(x), realValue(y));
    const auto q = round((realValue(x) - rem) / realValue(y)); // the integral quotient x/y, with zero derivative
    auto res = x - q * y;
    res[0] = rem; // the exact remainder, without the rounding errors of x - q*y
    return res;
}

template<typename X, typename Y, Requires<isRealOperable<X, Y>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto fmin(const X& x, const Y& y)
{
    return min(x, y);
}

template<typename X, typename Y, Requires<isRealOperable<X, Y>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto fmax(const X& x, const Y& y)
{
    return max(x, y);
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto floor(const Real<N, T>& x)
{
    Real<N, T> res; // all derivatives are zero
    res[0] = floor(x[0]);
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto ceil(const Real<N, T>& x)
{
    Real<N, T> res; // all derivatives are zero
    res[0] = ceil(x[0]);
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto round(const Real<N, T>& x)
{
    Real<N, T> res; // all derivatives are zero
    res[0] = round(x[0]);
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto trunc(const Real<N, T>& x)
{
    Real<N, T> res; // all derivatives are zero
    res[0] = trunc(x[0]);
    return res;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto sign(const Real<N, T>& x)
{
    Real<N, T> res; // all derivatives are zero
    res[0] = x[0] < T(0) ? T(-1) : (x[0] > T(0) ? T(1) : T(0));
    return res;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto copysign(const Real<N, T>& x, const U& y)
{
    return ((x[0] < T(0)) == (y < U(0))) ? x : -x; // |x| with the sign of y
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto copysign(const Real<N, T>& x, const Real<N, T>& y)
{
    return copysign(x, y[0]);
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto min(const Real<N, T>& x, const Real<N, T>& y)
{
    return (x[0] <= y[0]) ? x : y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto min(const Real<N, T>& x, const U& y)
{
    return (x[0] <= y) ? x : Real<N, T>{y};
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto min(const U& x, const Real<N, T>& y)
{
    return (x < y[0]) ? Real<N, T>{x} : y;
}

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto max(const Real<N, T>& x, const Real<N, T>& y)
{
    return (x[0] >= y[0]) ? x : y;
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto max(const Real<N, T>& x, const U& y)
{
    return (x[0] >= y) ? x : Real<N, T>{y};
}

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true>
AUTODIFF_DEVICE_FUNC constexpr auto max(const U& x, const Real<N, T>& y)
{
    return (x > y[0]) ? Real<N, T>{x} : y;
}

//=====================================================================================================================
//
// PRINTING FUNCTIONS
//
//=====================================================================================================================

#if AUTODIFF_HAS_IOSTREAM
template<size_t N, typename T>
std::ostream& operator<<(std::ostream& out, const Real<N, T>& x)
{
    out << x[0];
    return out;
}

template<size_t N, typename T>
auto repr(const Real<N, T>& x)
{
    std::stringstream ss;
    ss << "autodiff.real(";
    for(auto i = 0; i <= N; ++i)
        ss << (i == 0 ? "" : ", ") << x[i];
    ss << ")";
    return ss.str();
};
#endif // AUTODIFF_HAS_IOSTREAM

//=====================================================================================================================
//
// COMPARISON OPERATORS
//
//=====================================================================================================================

template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC bool operator==(const Real<N, T>& x, const Real<N, T>& y)
{
    bool res = true;
    For<0, N + 1>([&](auto i) constexpr {
        res = res && x[i] == y[i];
    });
    return res;
}

template<size_t N, typename T> AUTODIFF_DEVICE_FUNC bool operator!=(const Real<N, T>& x, const Real<N, T>& y) { return !(x == y); }
template<size_t N, typename T> AUTODIFF_DEVICE_FUNC bool operator< (const Real<N, T>& x, const Real<N, T>& y) { return x[0] <  y[0]; }
template<size_t N, typename T> AUTODIFF_DEVICE_FUNC bool operator> (const Real<N, T>& x, const Real<N, T>& y) { return x[0] >  y[0]; }
template<size_t N, typename T> AUTODIFF_DEVICE_FUNC bool operator<=(const Real<N, T>& x, const Real<N, T>& y) { return x[0] <= y[0]; }
template<size_t N, typename T> AUTODIFF_DEVICE_FUNC bool operator>=(const Real<N, T>& x, const Real<N, T>& y) { return x[0] >= y[0]; }

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator==(const Real<N, T>& x, const U& y) { return x[0] == y; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator!=(const Real<N, T>& x, const U& y) { return x[0] != y; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator< (const Real<N, T>& x, const U& y) { return x[0] <  y; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator> (const Real<N, T>& x, const U& y) { return x[0] >  y; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator<=(const Real<N, T>& x, const U& y) { return x[0] <= y; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator>=(const Real<N, T>& x, const U& y) { return x[0] >= y; }

template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator==(const U& x, const Real<N, T>& y) { return x == y[0]; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator!=(const U& x, const Real<N, T>& y) { return x != y[0]; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator< (const U& x, const Real<N, T>& y) { return x <  y[0]; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator> (const U& x, const Real<N, T>& y) { return x >  y[0]; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator<=(const U& x, const Real<N, T>& y) { return x <= y[0]; }
template<size_t N, typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC bool operator>=(const U& x, const Real<N, T>& y) { return x >= y[0]; }

//=====================================================================================================================
//
// SEED FUNCTION
//
//=====================================================================================================================

template<size_t order, size_t N, typename T, typename U>
AUTODIFF_DEVICE_FUNC auto seed(Real<N, T>& real, U&& seedval)
{
    static_assert(order == 1,
        "Real<N, T> is optimized for higher-order **directional** derivatives. "
        "You're possibly trying to use it for computing higher-order **cross** derivatives "
        "(e.g., `derivative(f, wrt(x, x, y), at(x, y))`) which is not supported by Real<N, T>. "
        "Use Dual<T, G> instead (e.g., `using dual4th = HigherOrderDual<4>;`)");
    real[order] = static_cast<T>(seedval);
}

/// Set the derivative of a real number so that its derivatives of all orders are the directional derivatives along `seedval`.
template<size_t N, typename T, typename U>
AUTODIFF_DEVICE_FUNC auto seedalong(Real<N, T>& real, U&& seedval)
{
    real[1] = static_cast<T>(seedval);
}

//=====================================================================================================================
//
// DERIVATIVE FUNCTIONS
//
//=====================================================================================================================

/// Return the value of a Real number.
template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto val(const Real<N, T>& x)
{
    return x[0];
}

/// Return the Real number itself: unlike dual numbers, operations on Real numbers do not produce expressions to be
/// evaluated. Provided for generic code also working with dual numbers (see autodiff issue #201).
template<size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto eval(const Real<N, T>& x)
{
    return x;
}

/// Return the derivative of a Real number with given order.
template<size_t order = 1, size_t N, typename T>
AUTODIFF_DEVICE_FUNC constexpr auto derivative(const Real<N, T>& x)
{
    return x[order];
}

//=====================================================================================================================
//
// NUMBER TRAITS DEFINITION
//
//=====================================================================================================================

template<size_t N, typename T>
struct NumberTraits<Real<N, T>>
{
    /// The underlying floating point type of Real<N, T>.
    using NumericType = T;

    /// The order of Real<N, T>.
    static constexpr auto Order = N;
};

} // namespace detail

//=====================================================================================================================
//
// CONVENIENT TYPE ALIASES
//
//=====================================================================================================================

using detail::Real;
using detail::val;
using detail::eval;
using detail::derivative;
#if AUTODIFF_HAS_IOSTREAM
using detail::repr;
#endif

using real0th = Real<0, double>;
using real1st = Real<1, double>;
using real2nd = Real<2, double>;
using real3rd = Real<3, double>;
using real4th = Real<4, double>;

using real = real1st;

} // namespace autodiff
