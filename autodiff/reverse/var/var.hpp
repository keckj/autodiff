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
#include <autodiff/common/stdlib.hpp>
#include <autodiff/common/memory.hpp>
#include <autodiff/common/meta.hpp>
#include <autodiff/common/numbertraits.hpp>
#include <autodiff/common/wrt.hpp>

// C++ includes
#if AUTODIFF_VAR_USE_STD_MEMORY
#include <functional>
#include <memory>
#endif
#if !AUTODIFF_NVRTC
#include <stdexcept>
#endif
#if AUTODIFF_HAS_IOSTREAM
#include <ostream>
#endif

/// autodiff namespace where @ref Variable and @ref grad are defined.
namespace autodiff {}

namespace autodiff {
// avoid clash with autodiff::detail in autodiff/forward/dual/dual.hpp
namespace reverse {
using detail::Requires;
using detail::For;
using detail::isArithmetic;
namespace detail {


using stdmath::abs;
using stdmath::acos;
using stdmath::asin;
using stdmath::atan;
using stdmath::atan2;
using stdmath::cos;
using stdmath::cosh;
using stdmath::erf;
using stdmath::exp;
using stdmath::hypot;
using stdmath::log;
using stdmath::log10;
using stdmath::pow;
using stdmath::sin;
using stdmath::sinh;
using stdmath::sqrt;
using stdmath::tan;
using stdmath::tanh;
using stdmath::asinh;
using stdmath::acosh;
using stdmath::atanh;
using stdmath::cbrt;
using stdmath::ceil;
using stdmath::copysign;
using stdmath::erfc;
using stdmath::exp2;
using stdmath::expm1;
using stdmath::floor;
using stdmath::fmax;
using stdmath::fmin;
using stdmath::fmod;
using stdmath::log1p;
using stdmath::log2;
using stdmath::round;
using stdmath::trunc;

template<typename T> struct Expr;
template<typename T> struct VariableExpr;
template<typename T> struct IndependentVariableExpr;
template<typename T> struct DependentVariableExpr;
template<typename T> struct ConstantExpr;
template<typename T> struct UnaryExpr;
template<typename T> struct NegativeExpr;
template<typename T> struct BinaryExpr;
template<typename T> struct TernaryExpr;
template<typename T> struct AddExpr;
template<typename T> struct SubExpr;
template<typename T> struct MulExpr;
template<typename T> struct DivExpr;
template<typename T> struct SinExpr;
template<typename T> struct CosExpr;
template<typename T> struct TanExpr;
template<typename T> struct SinhExpr;
template<typename T> struct CoshExpr;
template<typename T> struct TanhExpr;
template<typename T> struct ArcSinhExpr;
template<typename T> struct ArcCoshExpr;
template<typename T> struct ArcTanhExpr;
template<typename T> struct ArcSinExpr;
template<typename T> struct ArcCosExpr;
template<typename T> struct ArcTanExpr;
template<typename T> struct ArcTan2Expr;
template<typename T> struct ExpExpr;
template<typename T> struct LogExpr;
template<typename T> struct Log10Expr;
template<typename T> struct PowExpr;
template<typename T> struct SqrtExpr;
template<typename T> struct AbsExpr;
template<typename T> struct ErfExpr;
template<typename T> struct CbrtExpr;
template<typename T> struct Log2Expr;
template<typename T> struct Log1pExpr;
template<typename T> struct Exp2Expr;
template<typename T> struct Expm1Expr;
template<typename T> struct ErfcExpr;
template<typename T> struct FloorExpr;
template<typename T> struct CeilExpr;
template<typename T> struct RoundExpr;
template<typename T> struct TruncExpr;
template<typename T> struct FmodExpr;
template<typename T> struct Hypot2Expr;
template<typename T> struct Hypot3Expr;
template<typename T> struct Variable;

// Smart pointer and type erased function used to store expression trees (see AUTODIFF_VAR_USE_STD_MEMORY)
#if AUTODIFF_VAR_USE_STD_MEMORY
namespace memx = ::std;
#else
namespace memx = ::autodiff::detail;
#endif

template<typename T> using ExprPtr = memx::shared_ptr<Expr<T>>;

namespace traits {

template<typename T>
struct VariableValueTypeNotDefinedFor {};

template<typename T>
struct VariableValueType;

template<typename T>
struct VariableValueType { using type = stdx::conditional_t<isArithmetic<T>, T, VariableValueTypeNotDefinedFor<T>>; };

template<typename T>
struct VariableValueType<Variable<T>> { using type = typename VariableValueType<T>::type; };

template<typename T>
struct VariableValueType<ExprPtr<T>> { using type = typename VariableValueType<T>::type; };

template<typename T>
struct VariableOrder { constexpr static auto value = 0; };

template<typename T>
struct VariableOrder<Variable<T>> { constexpr static auto value = 1 + VariableOrder<T>::value; };

template<typename T>
struct isVariable { constexpr static bool value = false; };

template<typename T>
struct isVariable<Variable<T>> { constexpr static bool value = true; };

} // namespace traits

template<typename T>
using VariableValueType = typename traits::VariableValueType<T>::type;

template<typename T>
constexpr auto VariableOrder = traits::VariableOrder<T>::value;

template<typename T>
constexpr auto isVariable = traits::isVariable<T>::value;

/// The abstract type of any node type in the expression tree.
template<typename T>
struct Expr
{
    /// The value of this expression node.
    T val = {};

    /// Construct an Expr object with given value.
    AUTODIFF_DEVICE_FUNC explicit Expr(const T& v) : val(v) {}

    /// Destructor (to avoid warning)
    AUTODIFF_DEVICE_FUNC virtual ~Expr() {}

    /// Return true if this expression node is an independent variable
    AUTODIFF_DEVICE_FUNC virtual bool is_independent() const { return false; }

    /// Bind a value pointer for writing the derivative during propagation
    AUTODIFF_DEVICE_FUNC virtual void bind_value(T* /* grad */) {}

    /// Bind an expression pointer for writing the derivative expression during propagation
    AUTODIFF_DEVICE_FUNC virtual void bind_expr(ExprPtr<T>* /* gradx */) {}

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param wprime The derivative of the root expression node w.r.t. the child expression of this expression node.
    AUTODIFF_DEVICE_FUNC virtual void propagate(const T& wprime) = 0;

    /// Update the contribution of this expression in the derivative of the root node of the expression tree.
    /// @param wprime The derivative of the root expression node w.r.t. the child expression of this expression node (as an expression).
    AUTODIFF_DEVICE_FUNC virtual void propagatex(const ExprPtr<T>& wprime) = 0;

    /// Update the value of this expression
    AUTODIFF_DEVICE_FUNC virtual void update() = 0;
};

/// The node in the expression tree representing either an independent or dependent variable.
template<typename T>
struct VariableExpr : Expr<T>
{
    /// The derivative value of the root expression node w.r.t. this variable.
    T* gradPtr = {};

    /// The derivative expression of the root expression node w.r.t. this variable (reusable for higher-order derivatives).
    ExprPtr<T>* gradxPtr = {};

    /// Construct a VariableExpr object with given value.
    AUTODIFF_DEVICE_FUNC VariableExpr(const T& v) : Expr<T>(v) {}

    AUTODIFF_DEVICE_FUNC virtual void bind_value(T* grad) { gradPtr = grad; }
    AUTODIFF_DEVICE_FUNC virtual void bind_expr(ExprPtr<T>* gradx) { gradxPtr = gradx; }
};

/// The node in the expression tree representing an independent variable.
template<typename T>
struct IndependentVariableExpr : VariableExpr<T>
{
    using VariableExpr<T>::gradPtr;
    using VariableExpr<T>::gradxPtr;

    /// Construct an IndependentVariableExpr object with given value.
    AUTODIFF_DEVICE_FUNC IndependentVariableExpr(const T& v) : VariableExpr<T>(v) {}

    AUTODIFF_DEVICE_FUNC bool is_independent() const override { return true; }

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override {
        if(gradPtr) { *gradPtr += wprime; }
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        if(gradxPtr) { *gradxPtr = *gradxPtr + wprime; }
    }

    AUTODIFF_DEVICE_FUNC void update() override {}
};

/// The node in the expression tree representing a dependent variable.
template<typename T>
struct DependentVariableExpr : VariableExpr<T>
{
    using VariableExpr<T>::gradPtr;
    using VariableExpr<T>::gradxPtr;

    /// The expression tree that defines how the dependent variable is calculated.
    ExprPtr<T> expr;

    /// Construct an DependentVariableExpr object with given value.
    AUTODIFF_DEVICE_FUNC DependentVariableExpr(const ExprPtr<T>& e) : VariableExpr<T>(e->val), expr(e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        if(gradPtr) { *gradPtr += wprime; }
        expr->propagate(wprime);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        if(gradxPtr) { *gradxPtr = *gradxPtr + wprime; }
        expr->propagatex(wprime);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        expr->update();
        this->val = expr->val;
    }
};

template<typename T>
struct ConstantExpr : Expr<T>
{
    using Expr<T>::Expr;

    AUTODIFF_DEVICE_FUNC void propagate([[maybe_unused]] const T& wprime) override
    {}

    AUTODIFF_DEVICE_FUNC void propagatex([[maybe_unused]] const ExprPtr<T>& wprime) override
    {}

    AUTODIFF_DEVICE_FUNC void update() override {}
};

template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> constant(const T& val) { return memx::make_shared<ConstantExpr<T>>(val); }

template<typename T>
struct UnaryExpr : Expr<T>
{
    ExprPtr<T> x;

    AUTODIFF_DEVICE_FUNC UnaryExpr(const T& v, const ExprPtr<T>& e) : Expr<T>(v), x(e) {}
};

template<typename T>
struct NegativeExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    using UnaryExpr<T>::UnaryExpr;

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(-wprime);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(-wprime);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = -x->val;
    }
};

template<typename T>
struct BinaryExpr : Expr<T>
{
    ExprPtr<T> l, r;

    AUTODIFF_DEVICE_FUNC BinaryExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : Expr<T>(v), l(ll), r(rr) {}
};

template<typename T>
struct TernaryExpr : Expr<T>
{
    ExprPtr<T> l, c, r;

    AUTODIFF_DEVICE_FUNC TernaryExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& cc, const ExprPtr<T>& rr) : Expr<T>(v), l(ll), c(cc), r(rr) {}
};

template<typename T>
struct AddExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    using BinaryExpr<T>::BinaryExpr;

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        l->propagate(wprime);
        r->propagate(wprime);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime);
        r->propagatex(wprime);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        r->update();
        this->val = l->val + r->val;
    }
};

template<typename T>
struct SubExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        l->propagate(wprime);
        r->propagate(-wprime);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime);  // (l - r)'l =  l'
        r->propagatex(-wprime); // (l - r)'r = -r'
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        r->update();
        this->val = l->val - r->val;
    }
};

template<typename T>
struct MulExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        l->propagate(wprime * r->val); // (l * r)'l = w' * r
        r->propagate(wprime * l->val); // (l * r)'r = l * w'
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime * r);
        r->propagatex(wprime * l);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        r->update();
        this->val = l->val * r->val;
    }
};

template<typename T>
struct DivExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;
    using BinaryExpr<T>::BinaryExpr;

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        const auto aux1 = 1.0 / r->val;
        const auto aux2 = -l->val * aux1 * aux1;
        l->propagate(wprime * aux1);
        r->propagate(wprime * aux2);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux1 = 1.0 / r;
        const auto aux2 = -l * aux1 * aux1;
        l->propagatex(wprime * aux1);
        r->propagatex(wprime * aux2);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        r->update();
        this->val = l->val / r->val;
    }
};

template<typename T>
struct SinExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC SinExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime * cos(x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * cos(x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = sin(x->val);
    }
};

template<typename T>
struct CosExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC CosExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(-wprime * sin(x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(-wprime * sin(x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = cos(x->val);
    }
};

template<typename T>
struct TanExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC TanExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        const auto aux = 1.0 / cos(x->val);
        x->propagate(wprime * aux * aux);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = 1.0 / cos(x);
        x->propagatex(wprime * aux * aux);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = tan(x->val);
    }
};

template<typename T>
struct SinhExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC SinhExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime * cosh(x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * cosh(x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = sinh(x->val);
    }
};

template<typename T>
struct CoshExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC CoshExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime * sinh(x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * sinh(x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = cosh(x->val);
    }
};

template<typename T>
struct TanhExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC TanhExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        const auto aux = 1.0 / cosh(x->val);
        x->propagate(wprime * aux * aux);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = 1.0 / cosh(x);
        x->propagatex(wprime * aux * aux);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = tanh(x->val);
    }
};

template<typename T>
struct ArcSinhExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC ArcSinhExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / sqrt(x->val * x->val + 1.0));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / sqrt(x * x + 1.0));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = asinh(x->val);
    }
};

template<typename T>
struct ArcCoshExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC ArcCoshExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / sqrt(x->val * x->val - 1.0));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / sqrt(x * x - 1.0));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = acosh(x->val);
    }
};

template<typename T>
struct ArcTanhExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC ArcTanhExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / (1.0 - x->val * x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (1.0 - x * x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = atanh(x->val);
    }
};

template<typename T>
struct ArcSinExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC ArcSinExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / sqrt(1.0 - x->val * x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / sqrt(1.0 - x * x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = asin(x->val);
    }
};

template<typename T>
struct ArcCosExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC ArcCosExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(-wprime / sqrt(1.0 - x->val * x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(-wprime / sqrt(1.0 - x * x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = acos(x->val);
    }
};

template<typename T>
struct ArcTanExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC ArcTanExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / (1.0 + x->val * x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (1.0 + x * x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = atan(x->val);
    }
};

template<typename T>
struct ArcTan2Expr : BinaryExpr<T>
{
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    AUTODIFF_DEVICE_FUNC ArcTan2Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        const auto aux = wprime / (l->val * l->val + r->val * r->val);
        l->propagate(r->val * aux);
        r->propagate(-l->val * aux);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = wprime / (l * l + r * r);
        l->propagatex(r * aux);
        r->propagatex(-l * aux);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        r->update();
        this->val = atan2(l->val, r->val);
    }
};

template<typename T>
struct ExpExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::UnaryExpr;
    using UnaryExpr<T>::val;
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime * val); // exp(x)' = exp(x) * x'
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * exp(x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = exp(x->val);
    }
};

template<typename T>
struct LogExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using UnaryExpr<T>::UnaryExpr;

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / x->val); // log(x)' = x'/x
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / x);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = log(x->val);
    }
};

template<typename T>
struct Log10Expr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto ln10 = static_cast<VariableValueType<T>>(2.3025850929940456840179914546843);

    AUTODIFF_DEVICE_FUNC Log10Expr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / (ln10 * x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (ln10 * x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = log10(x->val);
    }
};

template<typename T>
struct PowExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    T log_l;

    AUTODIFF_DEVICE_FUNC PowExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr), log_l(log(ll->val)) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        using U = VariableValueType<T>;
        constexpr auto zero = U(0.0);
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * pow(lval, rval - 1);
        l->propagate(aux * rval);
        const auto auxr = lval == zero ? 0.0 : lval * log(lval); // since x*log(x) -> 0 as x -> 0
        r->propagate(aux * auxr);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        using U = VariableValueType<T>;
        constexpr auto zero = U(0.0);
        const auto aux = wprime * pow(l, r - 1);
        l->propagatex(aux * r);
        const auto auxr = l == zero ? 0.0*l : l * log(l); // since x*log(x) -> 0 as x -> 0
        r->propagatex(aux * auxr);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        r->update();
        this->val = pow(l->val, r->val);
    }
};

template<typename T>
struct PowConstantLeftExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    AUTODIFF_DEVICE_FUNC PowConstantLeftExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        const auto lval = l->val;
        const auto rval = r->val;
        const auto aux = wprime * pow(lval, rval - 1);
        const auto auxr = lval == 0.0 ? 0.0 : lval * log(lval); // since x*log(x) -> 0 as x -> 0
        r->propagate(aux * auxr);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = wprime * pow(l, r - 1);
        const auto auxr = l == 0.0 ? 0.0*l : l * log(l); // since x*log(x) -> 0 as x -> 0
        r->propagatex(aux * auxr);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        r->update();
        this->val = pow(l->val, r->val);
    }
};

template<typename T>
struct PowConstantRightExpr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    AUTODIFF_DEVICE_FUNC PowConstantRightExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        l->propagate(wprime * pow(l->val, r->val - 1) * r->val); // pow(l, r)'l = r * pow(l, r - 1) * l'
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime * pow(l, r - 1) * r);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        this->val = pow(l->val, r->val);
    }
};

template<typename T>
struct SqrtExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC SqrtExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / (2.0 * sqrt(x->val))); // sqrt(x)' = 1/2 * 1/sqrt(x) * x'
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (2.0 * sqrt(x)));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = sqrt(x->val);
    }
};

template<typename T>
struct AbsExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;
    using U = VariableValueType<T>;

    AUTODIFF_DEVICE_FUNC AbsExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        if(x->val < 0.0) x->propagate(-wprime);
        else if(x->val > 0.0) x->propagate(wprime);
        else x->propagate(T(0));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        if(x->val < 0.0) x->propagatex(-wprime);
        else if(x->val > 0.0) x->propagatex(wprime);
        else x->propagate(T(0));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = abs(x->val);
    }
};

template<typename T>
struct ErfExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto sqrt_pi = static_cast<VariableValueType<T>>(1.7724538509055160272981674833411451872554456638435);

    AUTODIFF_DEVICE_FUNC ErfExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        const auto aux = 2.0 / sqrt_pi * exp(-(x->val) * (x->val)); // erf(x)' = 2/sqrt(pi) * exp(-x * x) * x'
        x->propagate(wprime * aux);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        const auto aux = 2.0 / sqrt_pi * exp(-x * x);
        x->propagatex(wprime * aux);
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = erf(x->val);
    }
};

template<typename T>
struct CbrtExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC CbrtExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / (3.0 * cbrt(x->val) * cbrt(x->val)));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (3.0 * cbrt(x) * cbrt(x)));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = cbrt(x->val);
    }
};

template<typename T>
struct Log2Expr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto ln2 = static_cast<VariableValueType<T>>(0.69314718055994530941723212145818);

    AUTODIFF_DEVICE_FUNC Log2Expr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / (ln2 * x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (ln2 * x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = log2(x->val);
    }
};

template<typename T>
struct Log1pExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC Log1pExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime / (1.0 + x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime / (1.0 + x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = log1p(x->val);
    }
};

template<typename T>
struct Exp2Expr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto ln2 = static_cast<VariableValueType<T>>(0.69314718055994530941723212145818);

    AUTODIFF_DEVICE_FUNC Exp2Expr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime * ln2 * exp2(x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * ln2 * exp2(x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = exp2(x->val);
    }
};

template<typename T>
struct Expm1Expr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC Expm1Expr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(wprime * exp(x->val));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(wprime * exp(x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = expm1(x->val);
    }
};

template<typename T>
struct ErfcExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    constexpr static auto sqrt_pi = static_cast<VariableValueType<T>>(1.7724538509055160272981674833411451872554456638435);

    AUTODIFF_DEVICE_FUNC ErfcExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        x->propagate(-wprime * 2.0 / sqrt_pi * exp(-(x->val) * (x->val)));
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        x->propagatex(-wprime * 2.0 / sqrt_pi * exp(-x * x));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = erfc(x->val);
    }
};

template<typename T>
struct FloorExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC FloorExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate([[maybe_unused]] const T& wprime) override
    {} // floor(x)' = 0

    AUTODIFF_DEVICE_FUNC void propagatex([[maybe_unused]] const ExprPtr<T>& wprime) override
    {}

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = floor(x->val);
    }
};

template<typename T>
struct CeilExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC CeilExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate([[maybe_unused]] const T& wprime) override
    {} // ceil(x)' = 0

    AUTODIFF_DEVICE_FUNC void propagatex([[maybe_unused]] const ExprPtr<T>& wprime) override
    {}

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = ceil(x->val);
    }
};

template<typename T>
struct RoundExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC RoundExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate([[maybe_unused]] const T& wprime) override
    {} // round(x)' = 0

    AUTODIFF_DEVICE_FUNC void propagatex([[maybe_unused]] const ExprPtr<T>& wprime) override
    {}

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = round(x->val);
    }
};

template<typename T>
struct TruncExpr : UnaryExpr<T>
{
    // Using declarations for data members of base class
    using UnaryExpr<T>::x;

    AUTODIFF_DEVICE_FUNC TruncExpr(const T& v, const ExprPtr<T>& e) : UnaryExpr<T>(v, e) {}

    AUTODIFF_DEVICE_FUNC void propagate([[maybe_unused]] const T& wprime) override
    {} // trunc(x)' = 0

    AUTODIFF_DEVICE_FUNC void propagatex([[maybe_unused]] const ExprPtr<T>& wprime) override
    {}

    AUTODIFF_DEVICE_FUNC void update() override
    {
        x->update();
        this->val = trunc(x->val);
    }
};

template<typename T>
struct FmodExpr : BinaryExpr<T>
{
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    AUTODIFF_DEVICE_FUNC FmodExpr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    /// The integral quotient l/r, with zero derivative (fmod(l, r) = l - quotient * r).
    AUTODIFF_DEVICE_FUNC T quotient() const { return round((l->val - val) / r->val); }

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        l->propagate(wprime);
        r->propagate(-wprime * quotient());
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime);
        r->propagatex(-wprime * quotient());
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        r->update();
        this->val = fmod(l->val, r->val);
    }
};

template<typename T>
struct Hypot2Expr : BinaryExpr<T>
{
    // Using declarations for data members of base class
    using BinaryExpr<T>::val;
    using BinaryExpr<T>::l;
    using BinaryExpr<T>::r;

    AUTODIFF_DEVICE_FUNC Hypot2Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : BinaryExpr<T>(v, ll, rr) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        l->propagate(wprime * l->val / val); // sqrt(l*l + r*r)'l = 1/2 * 1/sqrt(l*l + r*r) * (2*l*l') = (l*l')/sqrt(l*l + r*r)
        r->propagate(wprime * r->val / val); // sqrt(l*l + r*r)'r = 1/2 * 1/sqrt(l*l + r*r) * (2*r*r') = (r*r')/sqrt(l*l + r*r)
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime * l / hypot(l, r));
        r->propagatex(wprime * r / hypot(l, r));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        r->update();
        this->val = hypot(l->val, r->val);
    }
};

/// The three-argument hypot of arithmetic values (std::hypot(x, y, z) is not available in device code) or variables.
template<typename T, typename X, typename Y, typename Z>
AUTODIFF_DEVICE_FUNC auto hypot3(const X& x, const Y& y, const Z& z)
{
    if constexpr(isArithmetic<T>)
        return stdmath::hypot3(static_cast<T>(x), static_cast<T>(y), static_cast<T>(z));
    else
        return hypot(x, y, z);
}

template<typename T>
struct Hypot3Expr : TernaryExpr<T>
{
    // Using declarations for data members of base class
    using TernaryExpr<T>::val;
    using TernaryExpr<T>::l;
    using TernaryExpr<T>::c;
    using TernaryExpr<T>::r;

    AUTODIFF_DEVICE_FUNC Hypot3Expr(const T& v, const ExprPtr<T>& ll, const ExprPtr<T>& cc, const ExprPtr<T>& rr) : TernaryExpr<T>(v, ll, cc, rr) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        l->propagate(wprime * l->val / val);
        c->propagate(wprime * c->val / val);
        r->propagate(wprime * r->val / val);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(wprime * l / hypot(l, c, r));
        c->propagatex(wprime * c / hypot(l, c, r));
        r->propagatex(wprime * r / hypot(l, c, r));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        l->update();
        c->update();
        r->update();
        this->val = hypot3<T>(l->val, c->val, r->val);
    }
};

// Any expression yielding a boolean depending on arithmetic subexpressions
struct BooleanExpr
{
    memx::function<bool()> expr;
    bool val = {};

    AUTODIFF_DEVICE_FUNC explicit BooleanExpr(memx::function<bool()> expression) : expr(stdx::move(expression)) { update(); }
    AUTODIFF_DEVICE_FUNC operator bool() const { return val; }

    AUTODIFF_DEVICE_FUNC void update() { val = expr(); }

    AUTODIFF_DEVICE_FUNC auto operator! () const { return BooleanExpr([e = expr]() { return !e(); }); } // capture the function, not this (which may be a temporary)
};

/// Capture numeric comparison between two expression trees
template<typename T, typename Comparator>
AUTODIFF_DEVICE_FUNC auto expr_comparison(const ExprPtr<T>& l, const ExprPtr<T>& r, Comparator&& compare) {
    return BooleanExpr([=]() mutable -> bool {
        l->update();
        r->update();
        return compare(l->val, r->val);
    });
}

template<typename Op> AUTODIFF_DEVICE_FUNC auto bool_expr_op(BooleanExpr& l, BooleanExpr& r, Op op) {
    return BooleanExpr([=]() mutable -> bool {
        l.update();
        r.update();
        return op(l, r);
    });
}

AUTODIFF_DEVICE_FUNC inline auto operator && (BooleanExpr&& l, BooleanExpr&& r) { return bool_expr_op(l, r, stdx::logical_and<> {}); }
AUTODIFF_DEVICE_FUNC inline auto operator || (BooleanExpr&& l, BooleanExpr&& r) { return bool_expr_op(l, r, stdx::logical_or<> {}); }

/// Select between expression branches depending on a boolean expression
template<typename T>
struct ConditionalExpr : Expr<T>
{
    // Using declarations for data members of base class
    BooleanExpr predicate;
    using Expr<T>::val;
    ExprPtr<T> l, r;

    AUTODIFF_DEVICE_FUNC ConditionalExpr(const BooleanExpr& wrappedPred, const ExprPtr<T>& ll, const ExprPtr<T>& rr) : Expr<T>(wrappedPred ? ll->val : rr->val), predicate(wrappedPred), l(ll), r(rr) {}

    AUTODIFF_DEVICE_FUNC void propagate(const T& wprime) override
    {
        if(predicate.val) l->propagate(wprime);
        else r->propagate(wprime);
    }

    AUTODIFF_DEVICE_FUNC void propagatex(const ExprPtr<T>& wprime) override
    {
        l->propagatex(derive(wprime, constant<T>(0.0)));
        r->propagatex(derive(constant<T>(0.0), wprime));
    }

    AUTODIFF_DEVICE_FUNC void update() override
    {
        predicate.update();
        if(predicate.val) {
            l->update();
            this->val = l->val;
        } else {
            r->update();
            this->val = r->val;
        }
    }

    AUTODIFF_DEVICE_FUNC ExprPtr<T> derive(const ExprPtr<T>& left, const ExprPtr<T>& right) const {
      return memx::make_shared<ConditionalExpr>(predicate, left, right);
    }
};

//------------------------------------------------------------------------------
// CONVENIENT FUNCTIONS
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const ExprPtr<T>& r) { return r; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const ExprPtr<T>& r) { return memx::make_shared<NegativeExpr<T>>(-r->val, r); }

template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const ExprPtr<T>& l, const ExprPtr<T>& r) { return memx::make_shared<AddExpr<T>>(l->val + r->val, l, r); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const ExprPtr<T>& l, const ExprPtr<T>& r) { return memx::make_shared<SubExpr<T>>(l->val - r->val, l, r); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator*(const ExprPtr<T>& l, const ExprPtr<T>& r) { return memx::make_shared<MulExpr<T>>(l->val * r->val, l, r); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator/(const ExprPtr<T>& l, const ExprPtr<T>& r) { return memx::make_shared<DivExpr<T>>(l->val / r->val, l, r); }

template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const U& l, const ExprPtr<T>& r) { return constant<T>(l) + r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const U& l, const ExprPtr<T>& r) { return constant<T>(l) - r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator*(const U& l, const ExprPtr<T>& r) { return constant<T>(l) * r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator/(const U& l, const ExprPtr<T>& r) { return constant<T>(l) / r; }

template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const ExprPtr<T>& l, const U& r) { return l + constant<T>(r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const ExprPtr<T>& l, const U& r) { return l - constant<T>(r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator*(const ExprPtr<T>& l, const U& r) { return l * constant<T>(r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator/(const ExprPtr<T>& l, const U& r) { return l / constant<T>(r); }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sin(const ExprPtr<T>& x) { return memx::make_shared<SinExpr<T>>(sin(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> cos(const ExprPtr<T>& x) { return memx::make_shared<CosExpr<T>>(cos(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> tan(const ExprPtr<T>& x) { return memx::make_shared<TanExpr<T>>(tan(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> asin(const ExprPtr<T>& x) { return memx::make_shared<ArcSinExpr<T>>(asin(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> acos(const ExprPtr<T>& x) { return memx::make_shared<ArcCosExpr<T>>(acos(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> atan(const ExprPtr<T>& x) { return memx::make_shared<ArcTanExpr<T>>(atan(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> atan2(const ExprPtr<T>& l, const ExprPtr<T>& r) { return memx::make_shared<ArcTan2Expr<T>>(atan2(l->val, r->val), l, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> atan2(const U& l, const ExprPtr<T>& r) { return memx::make_shared<ArcTan2Expr<T>>(atan2(l, r->val), constant<T>(l), r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> atan2(const ExprPtr<T>& l, const U& r) { return memx::make_shared<ArcTan2Expr<T>>(atan2(l->val, r), l, constant<T>(r)); }


//------------------------------------------------------------------------------
// HYPOT2 FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& r) { return memx::make_shared<Hypot2Expr<T>>(hypot(l->val, r->val), l, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const U& l, const ExprPtr<T>& r) { return memx::make_shared<Hypot2Expr<T>>(hypot(l, r->val), constant<T>(l), r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const ExprPtr<T>& l, const U& r) { return memx::make_shared<Hypot2Expr<T>>(hypot(l->val, r), l, constant<T>(r)); }

//------------------------------------------------------------------------------
// HYPOT3 FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& c, const ExprPtr<T>& r) { return memx::make_shared<Hypot3Expr<T>>(hypot3<T>(l->val,c->val, r->val), l, c, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const ExprPtr<T>& l, const ExprPtr<T>& c, const U& r) { return memx::make_shared<Hypot3Expr<T>>(hypot3<T>(l->val, c->val, r), l, c, constant<T>(r)); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const U& l, const ExprPtr<T>& c, const ExprPtr<T>& r) { return memx::make_shared<Hypot3Expr<T>>(hypot3<T>(l, c->val, r->val), constant<T>(l), c, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const ExprPtr<T>& l,const U& c, const ExprPtr<T>& r) { return memx::make_shared<Hypot3Expr<T>>(hypot3<T>(l->val, c, r->val), l, constant<T>(c), r); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const ExprPtr<T>& l, const U& c, const V& r) { return memx::make_shared<Hypot3Expr<T>>(hypot3<T>(l->val, c, r), l, constant<T>(c), constant<T>(r)); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const U& l, const ExprPtr<T>& c, const V& r) { return memx::make_shared<Hypot3Expr<T>>(hypot3<T>(l, c->val, r), constant<T>(l), c, constant<T>(r)); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const V& l, const U& c, const ExprPtr<T>& r) { return memx::make_shared<Hypot3Expr<T>>(hypot3<T>(l, c, r->val), constant<T>(l), constant<T>(c), r); }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sinh(const ExprPtr<T>& x) { return memx::make_shared<SinhExpr<T>>(sinh(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> cosh(const ExprPtr<T>& x) { return memx::make_shared<CoshExpr<T>>(cosh(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> tanh(const ExprPtr<T>& x) { return memx::make_shared<TanhExpr<T>>(tanh(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> asinh(const ExprPtr<T>& x) { return memx::make_shared<ArcSinhExpr<T>>(asinh(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> acosh(const ExprPtr<T>& x) { return memx::make_shared<ArcCoshExpr<T>>(acosh(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> atanh(const ExprPtr<T>& x) { return memx::make_shared<ArcTanhExpr<T>>(atanh(x->val), x); }

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> exp(const ExprPtr<T>& x) { return memx::make_shared<ExpExpr<T>>(exp(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> log(const ExprPtr<T>& x) { return memx::make_shared<LogExpr<T>>(log(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> log10(const ExprPtr<T>& x) { return memx::make_shared<Log10Expr<T>>(log10(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> log2(const ExprPtr<T>& x) { return memx::make_shared<Log2Expr<T>>(log2(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> log1p(const ExprPtr<T>& x) { return memx::make_shared<Log1pExpr<T>>(log1p(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> exp2(const ExprPtr<T>& x) { return memx::make_shared<Exp2Expr<T>>(exp2(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> expm1(const ExprPtr<T>& x) { return memx::make_shared<Expm1Expr<T>>(expm1(x->val), x); }

//------------------------------------------------------------------------------
// POWER FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sqrt(const ExprPtr<T>& x) { return memx::make_shared<SqrtExpr<T>>(sqrt(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> cbrt(const ExprPtr<T>& x) { return memx::make_shared<CbrtExpr<T>>(cbrt(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> pow(const ExprPtr<T>& l, const ExprPtr<T>& r) { return memx::make_shared<PowExpr<T>>(pow(l->val, r->val), l, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> pow(const U& l, const ExprPtr<T>& r) { return memx::make_shared<PowConstantLeftExpr<T>>(pow(l, r->val), constant<T>(l), r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> pow(const ExprPtr<T>& l, const U& r) { return memx::make_shared<PowConstantRightExpr<T>>(pow(l->val, r), l, constant<T>(r)); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> abs(const ExprPtr<T>& x) { return memx::make_shared<AbsExpr<T>>(abs(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> abs2(const ExprPtr<T>& x) { return x * x; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> conj(const ExprPtr<T>& x) { return x; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> real(const ExprPtr<T>& x) { return x; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> imag(const ExprPtr<T>&) { return constant<T>(0.0); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> erf(const ExprPtr<T>& x) { return memx::make_shared<ErfExpr<T>>(erf(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> erfc(const ExprPtr<T>& x) { return memx::make_shared<ErfcExpr<T>>(erfc(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> floor(const ExprPtr<T>& x) { return memx::make_shared<FloorExpr<T>>(floor(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> ceil(const ExprPtr<T>& x) { return memx::make_shared<CeilExpr<T>>(ceil(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> round(const ExprPtr<T>& x) { return memx::make_shared<RoundExpr<T>>(round(x->val), x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> trunc(const ExprPtr<T>& x) { return memx::make_shared<TruncExpr<T>>(trunc(x->val), x); }

/// The autodiff variable type used for detail mode automatic differentiation.
template<typename T>
struct Variable
{
    /// The pointer to the expression tree of variable operations
    ExprPtr<T> expr;

    /// Construct a default Variable object
    AUTODIFF_DEVICE_FUNC Variable() : Variable(0.0) {}

    /// Construct a copy of a Variable object
    AUTODIFF_DEVICE_FUNC Variable(const Variable& other) : Variable(other.expr) {}

    /// Construct a Variable object with given arithmetic value
    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC Variable(const U& val) : expr(memx::make_shared<IndependentVariableExpr<T>>(val)) {}

    /// Construct a Variable object with given expression
    AUTODIFF_DEVICE_FUNC Variable(const ExprPtr<T>& e) : expr(memx::make_shared<DependentVariableExpr<T>>(e)) {}

    /// Default copy assignment
    Variable& operator=(const Variable&) = default;

    /// Update the value of this variable with changes in its expression tree
    AUTODIFF_DEVICE_FUNC void update() { expr->update(); }

    AUTODIFF_DEVICE_FUNC void update(T value) {
      if(expr->is_independent()) {
        expr->val = value;
        expr->update();
      } else {
        AUTODIFF_THROW(std::logic_error, "Cannot update the value of a dependent expression stored in a variable");
      }
    }

    /// Convert this Variable object into an expression pointer.
    /// Explicit, since ExprPtr<T> converts implicitly into Variable: two way implicit conversions make expressions such as
    /// `cond ? var : ExprPtr` ambiguous (e.g. in Eigen 3.4 triangular solvers, see autodiff issue #141).
    AUTODIFF_DEVICE_FUNC explicit operator const ExprPtr<T>&() const { return expr; }

    /// Assign an arithmetic value to this variable.
    template<typename U, Requires<isArithmetic<U>> = true>
    AUTODIFF_DEVICE_FUNC auto operator=(const U& val) -> Variable& { *this = Variable(val); return *this; }

    /// Assign an expression to this variable.
    AUTODIFF_DEVICE_FUNC auto operator=(const ExprPtr<T>& x) -> Variable& { *this = Variable(x); return *this; }

    // Assignment operators
    AUTODIFF_DEVICE_FUNC Variable& operator+=(const ExprPtr<T>& x) { *this = Variable(expr + x); return *this; }
    AUTODIFF_DEVICE_FUNC Variable& operator-=(const ExprPtr<T>& x) { *this = Variable(expr - x); return *this; }
    AUTODIFF_DEVICE_FUNC Variable& operator*=(const ExprPtr<T>& x) { *this = Variable(expr * x); return *this; }
    AUTODIFF_DEVICE_FUNC Variable& operator/=(const ExprPtr<T>& x) { *this = Variable(expr / x); return *this; }
    AUTODIFF_DEVICE_FUNC Variable& operator+=(const Variable& x) { return *this += x.expr; }
    AUTODIFF_DEVICE_FUNC Variable& operator-=(const Variable& x) { return *this -= x.expr; }
    AUTODIFF_DEVICE_FUNC Variable& operator*=(const Variable& x) { return *this *= x.expr; }
    AUTODIFF_DEVICE_FUNC Variable& operator/=(const Variable& x) { return *this /= x.expr; }

	// Assignment operators with arithmetic values
    template<typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC Variable& operator+=(const U& x) { *this = Variable(expr + x); return *this; }
    template<typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC Variable& operator-=(const U& x) { *this = Variable(expr - x); return *this; }
    template<typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC Variable& operator*=(const U& x) { *this = Variable(expr * x); return *this; }
    template<typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC Variable& operator/=(const U& x) { *this = Variable(expr / x); return *this; }

#if defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION_VAR) || defined(AUTODIFF_ENABLE_IMPLICIT_CONVERSION)
    AUTODIFF_DEVICE_FUNC operator T() const { return expr->val; }

    template<typename U>
    AUTODIFF_DEVICE_FUNC operator U() const { return static_cast<U>(expr->val); }
#else
    AUTODIFF_DEVICE_FUNC explicit operator T() const { return expr->val; }

    template<typename U>
    AUTODIFF_DEVICE_FUNC explicit operator U() const { return static_cast<U>(expr->val); }
#endif
};

//------------------------------------------------------------------------------
// EXPRESSION TRAITS
//------------------------------------------------------------------------------

template<typename T, Requires<isArithmetic<T>> = true> AUTODIFF_DEVICE_FUNC T expr_value(const T& t) { return t; }
template<typename T> AUTODIFF_DEVICE_FUNC T expr_value(const ExprPtr<T>& t) { return t->val; }
template<typename T> AUTODIFF_DEVICE_FUNC T expr_value(const Variable<T>& t) { return t.expr->val; }

template<typename T, typename U>
using expr_common_t = stdx::common_type_t<decltype(expr_value(stdx::declval<T>())), decltype(expr_value(stdx::declval<U>()))>;

template<class> struct sfinae_true : stdx::true_type {};
template<typename T> static auto is_expr_test(int) -> sfinae_true<decltype(expr_value(stdx::declval<T>()))>;
template<typename T> static auto is_expr_test(long) -> stdx::false_type;
template<typename T> struct is_expr : decltype(is_expr_test<T>(0)) {};
template<typename T> constexpr bool is_expr_v = is_expr<T>::value;

template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> coerce_expr(const U& u) { return constant<T>(u); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> coerce_expr(const ExprPtr<T>& t) { return t; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> coerce_expr(const Variable<T>& t) { return t.expr; }

template<typename T, typename U> struct is_binary_expr : stdx::conditional_t<!(isArithmetic<T> && isArithmetic<U>) && is_expr_v<T> && is_expr_v<U>, stdx::true_type, stdx::false_type> {};
template<typename T, typename U> constexpr bool is_binary_expr_v = is_binary_expr<T, U>::value;


//------------------------------------------------------------------------------
// COMPARISON OPERATORS
//------------------------------------------------------------------------------

template<typename Comparator, typename T, typename U>
AUTODIFF_DEVICE_FUNC auto comparison_operator(const T& t, const U& u) {
    using C = expr_common_t<T, U>;
    return expr_comparison(coerce_expr<C>(t), coerce_expr<C>(u), Comparator {});
}

template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
AUTODIFF_DEVICE_FUNC auto operator == (const T& t, const U& u) { return comparison_operator<stdx::equal_to<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
AUTODIFF_DEVICE_FUNC auto operator != (const T& t, const U& u) { return comparison_operator<stdx::not_equal_to<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
AUTODIFF_DEVICE_FUNC auto operator <= (const T& t, const U& u) { return comparison_operator<stdx::less_equal<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
AUTODIFF_DEVICE_FUNC auto operator >= (const T& t, const U& u) { return comparison_operator<stdx::greater_equal<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
AUTODIFF_DEVICE_FUNC auto operator < (const T& t, const U& u) { return comparison_operator<stdx::less<>>(t, u); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
AUTODIFF_DEVICE_FUNC auto operator > (const T& t, const U& u) { return comparison_operator<stdx::greater<>>(t, u); }

//------------------------------------------------------------------------------
// CONDITION AND RELATED FUNCTIONS
//------------------------------------------------------------------------------

template<typename T, typename U, Requires<is_expr_v<T> && is_expr_v<U>> = true>
AUTODIFF_DEVICE_FUNC auto condition(BooleanExpr&& p, const T& t, const U& u) {
  using C = expr_common_t<T, U>;
  ExprPtr<C> expr = memx::make_shared<ConditionalExpr<C>>(stdx::forward<BooleanExpr>(p), coerce_expr<C>(t), coerce_expr<C>(u));
  return expr;
}

template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true> AUTODIFF_DEVICE_FUNC auto min(const T& x, const U& y) { return condition(x < y, x, y); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true> AUTODIFF_DEVICE_FUNC auto max(const T& x, const U& y) { return condition(x > y, x, y); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sgn(const ExprPtr<T>& x) { return condition(x < 0, -1.0, condition(x > 0, 1.0, 0.0)); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sgn(const Variable<T>& x) { return condition(x.expr < 0, -1.0, condition(x.expr > 0, 1.0, 0.0)); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sign(const ExprPtr<T>& x) { return sgn(x); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sign(const Variable<T>& x) { return sgn(x); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true> AUTODIFF_DEVICE_FUNC auto fmin(const T& x, const U& y) { return min(x, y); }
template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true> AUTODIFF_DEVICE_FUNC auto fmax(const T& x, const U& y) { return max(x, y); }

template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
AUTODIFF_DEVICE_FUNC auto copysign(const T& x, const U& y)
{
    using C = expr_common_t<T, U>;
    const ExprPtr<C> mag = abs(coerce_expr<C>(x));
    return condition(coerce_expr<C>(y) < 0, -mag, mag); // |x| with the sign of y
}

template<typename T, typename U, Requires<is_binary_expr_v<T, U>> = true>
AUTODIFF_DEVICE_FUNC auto fmod(const T& x, const U& y)
{
    using C = expr_common_t<T, U>;
    const ExprPtr<C> l = coerce_expr<C>(x);
    const ExprPtr<C> r = coerce_expr<C>(y);
    ExprPtr<C> expr = memx::make_shared<FmodExpr<C>>(fmod(l->val, r->val), l, r);
    return expr;
}

//------------------------------------------------------------------------------
// ARITHMETIC OPERATORS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC const ExprPtr<T>& operator+(const Variable<T>& r) { return r.expr; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const Variable<T>& r) { return -r.expr; }

template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const Variable<T>& l, const Variable<T>& r) { return l.expr + r.expr; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const Variable<T>& l, const Variable<T>& r) { return l.expr - r.expr; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator*(const Variable<T>& l, const Variable<T>& r) { return l.expr * r.expr; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator/(const Variable<T>& l, const Variable<T>& r) { return l.expr / r.expr; }

template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const ExprPtr<T>& l, const Variable<T>& r) { return l + r.expr; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const ExprPtr<T>& l, const Variable<T>& r) { return l - r.expr; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator*(const ExprPtr<T>& l, const Variable<T>& r) { return l * r.expr; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator/(const ExprPtr<T>& l, const Variable<T>& r) { return l / r.expr; }

template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr + r; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr - r; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator*(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr * r; }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator/(const Variable<T>& l, const ExprPtr<T>& r) { return l.expr / r; }

template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const U& l, const Variable<T>& r) { return l + r.expr; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const U& l, const Variable<T>& r) { return l - r.expr; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator*(const U& l, const Variable<T>& r) { return l * r.expr; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator/(const U& l, const Variable<T>& r) { return l / r.expr; }

template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator+(const Variable<T>& l, const U& r) { return l.expr + r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator-(const Variable<T>& l, const U& r) { return l.expr - r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator*(const Variable<T>& l, const U& r) { return l.expr * r; }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> operator/(const Variable<T>& l, const U& r) { return l.expr / r; }

//------------------------------------------------------------------------------
// TRIGONOMETRIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sin(const Variable<T>& x) { return sin(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> cos(const Variable<T>& x) { return cos(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> tan(const Variable<T>& x) { return tan(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> asin(const Variable<T>& x) { return asin(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> acos(const Variable<T>& x) { return acos(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> atan(const Variable<T>& x) { return atan(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> atan2(const Variable<T> & l, const Variable<T> & r) { return atan2(l.expr, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> atan2(const U& l, const Variable<T>& r) { return atan2(l, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> atan2(const Variable<T>& l, const U& r) { return atan2(l.expr, r); }

//------------------------------------------------------------------------------
// HYPOT2 FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const Variable<T>& l, const Variable<T>& r) { return hypot(l.expr, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const U& l, const Variable<T>& r) { return hypot(l, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const Variable<T>& l, const U& r) { return hypot(l.expr, r); }

//------------------------------------------------------------------------------
// HYPOT3 FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const Variable<T> &l, const Variable<T> &c, const Variable<T> &r) { return hypot(l.expr, c.expr, r.expr); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const Variable<T>& l, const U& c, const V& r) { return hypot(l.expr, c, r); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const U& l, const Variable<T>& c, const V& r) { return hypot(l, c.expr, r); }
template<typename T, typename U, typename V, Requires<isArithmetic<U> && isArithmetic<V>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const U& l, const V& c, const Variable<T>& r) { return hypot(l, c, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const Variable<T> &l, const Variable<T> &c, const U& r) { return hypot(l.expr, c.expr, r); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const U &l, const Variable<T> &c, const Variable<T>& r) { return hypot(l, c.expr, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> hypot(const Variable<T> &l, const U &c, const Variable<T>& r) { return hypot(l.expr, c, r.expr); }

//------------------------------------------------------------------------------
// HYPERBOLIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sinh(const Variable<T>& x) { return sinh(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> cosh(const Variable<T>& x) { return cosh(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> tanh(const Variable<T>& x) { return tanh(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> asinh(const Variable<T>& x) { return asinh(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> acosh(const Variable<T>& x) { return acosh(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> atanh(const Variable<T>& x) { return atanh(x.expr); }

//------------------------------------------------------------------------------
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> exp(const Variable<T>& x) { return exp(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> log(const Variable<T>& x) { return log(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> log10(const Variable<T>& x) { return log10(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> log2(const Variable<T>& x) { return log2(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> log1p(const Variable<T>& x) { return log1p(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> exp2(const Variable<T>& x) { return exp2(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> expm1(const Variable<T>& x) { return expm1(x.expr); }

//------------------------------------------------------------------------------
// POWER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> sqrt(const Variable<T>& x) { return sqrt(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> cbrt(const Variable<T>& x) { return cbrt(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> pow(const Variable<T>& l, const Variable<T>& r) { return pow(l.expr, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> pow(const U& l, const Variable<T>& r) { return pow(l, r.expr); }
template<typename T, typename U, Requires<isArithmetic<U>> = true> AUTODIFF_DEVICE_FUNC ExprPtr<T> pow(const Variable<T>& l, const U& r) { return pow(l.expr, r); }

//------------------------------------------------------------------------------
// OTHER FUNCTIONS (DEFINED FOR ARGUMENTS OF TYPE Variable)
//------------------------------------------------------------------------------
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> abs(const Variable<T>& x) { return abs(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> abs2(const Variable<T>& x) { return abs2(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> conj(const Variable<T>& x) { return conj(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> real(const Variable<T>& x) { return real(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> imag(const Variable<T>& x) { return imag(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> erf(const Variable<T>& x) { return erf(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> erfc(const Variable<T>& x) { return erfc(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> floor(const Variable<T>& x) { return floor(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> ceil(const Variable<T>& x) { return ceil(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> round(const Variable<T>& x) { return round(x.expr); }
template<typename T> AUTODIFF_DEVICE_FUNC ExprPtr<T> trunc(const Variable<T>& x) { return trunc(x.expr); }

template<typename T, Requires<is_expr_v<T>> = true>
AUTODIFF_DEVICE_FUNC auto val(const T& t) { return expr_value(t); }

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename T>
[[deprecated("Use method `derivatives(y, wrt(a, b, c,...)` instead.")]]
AUTODIFF_DEVICE_FUNC auto derivatives(const T&)
{
    static_assert(!stdx::is_same_v<T,T>, "Method derivatives(const var&) has been deprecated. Use method derivatives(y, wrt(a, b, c,...) instead.");
}

/// Return the derivatives of a variable y with respect to all independent variables.
template<typename T>
[[deprecated("Use method derivativesx(y, wrt(a, b, c,...) instead.")]]
AUTODIFF_DEVICE_FUNC auto derivativesx(const T&)
{
    static_assert(!stdx::is_same_v<T,T>, "Method derivativesx(const var&) has been deprecated. Use method derivativesx(y, wrt(a, b, c,...) instead.");
}

// The same Wrt as in forward mode, so that forward and reverse modes can be used together
using ::autodiff::detail::Wrt;
using ::autodiff::detail::wrt;

/// Return the derivatives of a dependent variable y with respect given independent variables.
template<typename T, typename... Vars>
AUTODIFF_DEVICE_FUNC auto derivatives(const Variable<T>& y, const Wrt<Vars...>& wrt)
{
    constexpr auto N = sizeof...(Vars);
    stdx::array<T, N> values = {}; // value initialized (std::array::fill is not usable in device code with nvcc)

    For<N>([&](auto i) constexpr {
        stdx::get<i>(wrt.args).expr->bind_value(&values[i]);
    });

    y.expr->propagate(1.0);

    For<N>([&](auto i) constexpr {
        stdx::get<i>(wrt.args).expr->bind_value(nullptr);
    });

    return values;
}

/// Return the derivatives of a dependent variable y with respect given independent variables.
template<typename T, typename... Vars>
AUTODIFF_DEVICE_FUNC auto derivativesx(const Variable<T>& y, const Wrt<Vars...>& wrt)
{
    constexpr auto N = sizeof...(Vars);
    stdx::array<Variable<T>, N> values;

    For<N>([&](auto i) constexpr {
        stdx::get<i>(wrt.args).expr->bind_expr(&values[i].expr);
    });

    y.expr->propagatex(constant<T>(1.0));

    For<N>([&](auto i) constexpr {
        stdx::get<i>(wrt.args).expr->bind_expr(nullptr);
    });

    return values;
}

#if AUTODIFF_HAS_IOSTREAM
/// Output a Variable object to the output stream.
template<typename T>
std::ostream& operator<<(std::ostream& out, const Variable<T>& x)
{
    out << val(x);
    return out;
}

/// Output an ExprPrt object to the output stream.
template<typename T>
std::ostream& operator<<(std::ostream& out, const ExprPtr<T>& x)
{
    out << val(x);
    return out;
}
#endif

//=====================================================================================================================
//
// HIGHER-ORDER VAR NUMBERS
//
//=====================================================================================================================

template<size_t N, typename T>
struct AuxHigherOrderVariable;

template<typename T>
struct AuxHigherOrderVariable<0, T>
{
    using type = T;
};

template<size_t N, typename T>
struct AuxHigherOrderVariable
{
    using type = Variable<typename AuxHigherOrderVariable<N - 1, T>::type>;
};

template<size_t N, typename T>
using HigherOrderVariable = typename AuxHigherOrderVariable<N, T>::type;

} // namespace detail

} // namespace reverse

using reverse::detail::wrt;
using reverse::detail::derivatives;
using reverse::detail::Variable;
using reverse::detail::val;

using var = Variable<double>;

AUTODIFF_DEVICE_FUNC inline reverse::detail::BooleanExpr boolref(const bool& v) { return reverse::detail::BooleanExpr([&]() { return v; }); }

} // namespace autodiff
