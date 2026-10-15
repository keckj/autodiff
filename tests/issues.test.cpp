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

// Regression tests of bugs reported upstream (https://github.com/autodiff/autodiff/issues). The forward and reverse
// modes are deliberately included together (issue #179).

// C++ includes
#include <cmath>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/utils/taylorseries.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
#include <tests/utils/catch.hpp>

// Eigen includes
#include <Eigen/LU>
using namespace autodiff;

namespace {

template<typename X>
auto fsinx(const X& x) -> X { return sin(x) * x; }

template<typename X>
auto fxy(const X& x, const X& y) -> X { return x * y * sin(x) + exp(y); }

} // namespace

TEST_CASE("testing forward and reverse modes together (#179)", "[issues][dual][real][var]")
{
    dual x = 2.0;
    const double dfdx = derivative([](dual x) -> dual { return x * x; }, wrt(x), at(x));
    CHECK_APPROX( dfdx, 4.0 );

    var v = 3.0;
    var y = v * v;
    const auto [dydv] = derivatives(y, wrt(v));
    CHECK_APPROX( dydv, 6.0 );

    Vector3var xv(1.0, 2.0, 3.0);
    var z = xv.sum() * xv[0];
    const Eigen::VectorXd gv = gradient(z, xv);
    Vector3dual xd(1.0, 2.0, 3.0);
    const auto gd = gradient([](const Vector3dual& x) -> dual { return x.sum() * x[0]; }, wrt(xd), at(xd));
    CHECK_APPROX( gv[0], 7.0 );
    CHECK_APPROX( gd[0], 7.0 );
    CHECK_APPROX( gv[2], gd[2] );
}

TEST_CASE("testing val and eval of real numbers with dual numbers included (#201)", "[issues][dual][real]")
{
    real x = 2.0;
    const real cx = 3.0;
    dual y = 4.0;
    var z = 5.0;
    CHECK( val(x) == 2.0 );
    CHECK( val(cx) == 3.0 );
    CHECK( val(y) == 4.0 );
    CHECK( val(z) == 5.0 );
    CHECK( val(x * cx) == 6.0 );

    // eval is a no-op on real numbers, so that generic code can also use it with dual numbers
    real4th w = 1.5;
    w[1] = 2.0;
    const auto ew = eval(w);
    static_assert(std::is_same_v<std::remove_cv_t<decltype(ew)>, real4th>);
    CHECK( ew[0] == 1.5 );
    CHECK( ew[1] == 2.0 );
    CHECK( val(eval(y * y)) == 16.0 );
}

TEST_CASE("testing higher-order derivatives of real numbers with respect to one variable (#201, #386)", "[issues][real]")
{
    real4th x = 1.0;
    dual4th xd = 1.0;
    const auto d = derivatives(fsinx<real4th>, wrt(x), at(x));
    const auto dd = derivatives(fsinx<dual4th>, wrt(xd), at(xd));
    const auto da = derivatives(fsinx<real4th>, along(1.0), at(x));
    const double s = std::sin(1.0), c = std::cos(1.0);
    const double exact[] = { s, c + s, 2 * c - s, -3 * s - c, -4 * c + s };
    for(auto i = 0; i < 5; ++i) {
        CHECK_APPROX( d[i], exact[i] );
        CHECK_APPROX( dd[i], exact[i] );
        CHECK_APPROX( da[i], exact[i] );
    }
    CHECK_APPROX( derivative<3>(fsinx<real4th>, wrt(x), at(x)), exact[3] );

    // The seeding is undone after the evaluation
    CHECK( x[1] == 0.0 );

    // Gradients of higher-order real numbers
    VectorXreal2nd v(2);
    v << 1.0, 2.0;
    const Eigen::VectorXd g = gradient([](const VectorXreal2nd& v) -> real2nd { return v[0] * v[0] * v[1]; }, wrt(v), at(v));
    CHECK_APPROX( g[0], 4.0 );
    CHECK_APPROX( g[1], 1.0 );
}

TEST_CASE("testing higher-order directional derivatives of dual numbers", "[issues][dual]")
{
    // dual numbers were seeded along the direction only at 1st order: the derivatives of higher orders were zero
    dual4th x = 1.0;
    real4th xr = 1.0;
    const auto d = derivatives(fsinx<dual4th>, along(2.0), at(x));
    const auto dr = derivatives(fsinx<real4th>, along(2.0), at(xr));
    for(auto i = 0; i < 5; ++i)
        CHECK_APPROX( d[i], dr[i] );
    CHECK_APPROX( d[2], 4 * (2 * std::cos(1.0) - std::sin(1.0)) );
    CHECK( val(x.val.val.grad) == 0.0 ); // unseeded

    dual4th u = 0.5, w = 1.5;
    real4th ur = 0.5, wr = 1.5;
    const auto e = derivatives(fxy<dual4th>, along(2.0, -1.0), at(u, w));
    const auto er = derivatives(fxy<real4th>, along(2.0, -1.0), at(ur, wr));
    for(auto i = 0; i < 5; ++i)
        CHECK_APPROX( e[i], er[i] );

    VectorXdual3rd p(2);
    VectorXreal3rd pr(2);
    p << 0.5, 1.5;
    pr << 0.5, 1.5;
    const Eigen::Vector2d dir(1.0, 3.0);
    const auto q = derivatives([](const VectorXdual3rd& p) -> dual3rd { return fxy(p[0], p[1]); }, along(dir), at(p));
    const auto qr = derivatives([](const VectorXreal3rd& p) -> real3rd { return fxy(p[0], p[1]); }, along(dir), at(pr));
    for(auto i = 0; i < 4; ++i)
        CHECK_APPROX( q[i], qr[i] );

    // Taylor series of dual numbers
    auto gd = taylorseries(fsinx<dual4th>, along(1.0), at(x));
    auto gr = taylorseries(fsinx<real4th>, along(1.0), at(xr));
    CHECK_APPROX( gd(0.5), gr(0.5) );
    CHECK( std::abs(gd(0.1) - std::sin(1.1) * 1.1) < 1e-6 );
}

TEST_CASE("testing chain rule of inverse hyperbolic functions of real numbers (#395, #396)", "[issues][real]")
{
    real x = 1.0;
    CHECK_APPROX( derivative([](real x) -> real { return asinh(tan(x)); }, wrt(x), at(x)), 1.0 / std::cos(1.0) );
    x = 0.1;
    CHECK_APPROX( derivative([](real x) -> real { return atanh(2.0 * x); }, wrt(x), at(x)), 2.0 / (1.0 - 0.04) );
    x = 1.0;
    CHECK_APPROX( derivative([](real x) -> real { return acosh(2.0 * x + 1.0); }, wrt(x), at(x)), 2.0 / std::sqrt(8.0) );

    // Higher orders compared with the logarithmic expressions of the inverse hyperbolic functions
    real4th y = 0.3;
    const auto a = derivatives([](real4th x) -> real4th { return asinh(tan(x)); }, wrt(y), at(y));
    const auto al = derivatives([](real4th x) -> real4th { const real4th u = tan(x); return log(u + sqrt(u * u + 1)); }, wrt(y), at(y));
    const auto b = derivatives([](real4th x) -> real4th { return acosh(2.0 * x * x + 1.0); }, wrt(y), at(y));
    const auto bl = derivatives([](real4th x) -> real4th { const real4th u = 2.0 * x * x + 1.0; return log(u + sqrt(u * u - 1)); }, wrt(y), at(y));
    const auto c = derivatives([](real4th x) -> real4th { return atanh(2.0 * x); }, wrt(y), at(y));
    const auto cl = derivatives([](real4th x) -> real4th { const real4th u = 2.0 * x; return 0.5 * log((1 + u) / (1 - u)); }, wrt(y), at(y));
    for(auto i = 0; i < 5; ++i) {
        CHECK( a[i] == Catch::Approx(al[i]).epsilon(1e-10) );
        CHECK( b[i] == Catch::Approx(bl[i]).epsilon(1e-10) );
        CHECK( c[i] == Catch::Approx(cl[i]).epsilon(1e-10) );
    }
}

TEST_CASE("testing inverse hyperbolic functions of dual and var numbers (missing upstream)", "[issues][dual][var]")
{
    // Higher orders compared with the logarithmic expressions of the inverse hyperbolic functions
    dual4th y = 0.3;
    const auto a = derivatives([](dual4th x) -> dual4th { return asinh(tan(x)); }, wrt(y), at(y));
    const auto al = derivatives([](dual4th x) -> dual4th { const dual4th u = tan(x); return log(u + sqrt(u * u + 1)); }, wrt(y), at(y));
    const auto b = derivatives([](dual4th x) -> dual4th { return acosh(2.0 * x * x + 1.0); }, wrt(y), at(y));
    const auto bl = derivatives([](dual4th x) -> dual4th { const dual4th u = 2.0 * x * x + 1.0; return log(u + sqrt(u * u - 1)); }, wrt(y), at(y));
    const auto c = derivatives([](dual4th x) -> dual4th { return atanh(2.0 * x); }, wrt(y), at(y));
    const auto cl = derivatives([](dual4th x) -> dual4th { const dual4th u = 2.0 * x; return 0.5 * log((1 + u) / (1 - u)); }, wrt(y), at(y));
    for(auto i = 0; i < 5; ++i) {
        CHECK( a[i] == Catch::Approx(al[i]).epsilon(1e-10) );
        CHECK( b[i] == Catch::Approx(bl[i]).epsilon(1e-10) );
        CHECK( c[i] == Catch::Approx(cl[i]).epsilon(1e-10) );
    }

    // Second order derivatives of var
    var x = 0.3;
    const var u = tan(x);
    const var v = 2.0 * x * x + 1.0;
    const var w = 2.0 * x;
    const var fs[] = { asinh(u), acosh(v), atanh(w) };
    const var gs[] = { log(u + sqrt(u * u + 1)), log(v + sqrt(v * v - 1)), 0.5 * log((1 + w) / (1 - w)) };
    for(auto i = 0; i < 3; ++i) {
        CHECK_APPROX( val(fs[i]), val(gs[i]) );
        const auto [dfdx] = derivativesx(fs[i], wrt(x));
        const auto [dgdx] = derivativesx(gs[i], wrt(x));
        CHECK_APPROX( val(dfdx), val(dgdx) );
        const auto [d2fdx2] = derivatives(dfdx, wrt(x));
        const auto [d2gdx2] = derivatives(dgdx, wrt(x));
        CHECK_APPROX( d2fdx2, d2gdx2 );
    }
}

TEST_CASE("testing math functions of dual, real and var numbers (missing upstream)", "[issues][dual][real][var]")
{
    // Higher orders of dual4th and real4th compared with equivalent expressions and with each other
    const auto check = [](auto f, auto g) {
        dual4th xd = 0.3;
        real4th xr = 0.3;
        const auto fd = derivatives(f, wrt(xd), at(xd));
        const auto gd = derivatives(g, wrt(xd), at(xd));
        const auto fr = derivatives(f, wrt(xr), at(xr));
        const auto gr = derivatives(g, wrt(xr), at(xr));
        for(auto i = 0; i < 5; ++i) {
            CHECK( fd[i] == Catch::Approx(gd[i]).epsilon(1e-10).margin(1e-12) );
            CHECK( fr[i] == Catch::Approx(gr[i]).epsilon(1e-10).margin(1e-12) );
            CHECK( fr[i] == Catch::Approx(fd[i]).epsilon(1e-10).margin(1e-12) );
        }
    };
    #define FN(expr) [](auto x) -> decltype(x) { return expr; }
    check(FN(log2(x)), FN(log(x) / log(2.0)));
    check(FN(log1p(x)), FN(log(1 + x)));
    check(FN(exp2(x)), FN(exp(x * log(2.0))));
    check(FN(expm1(x)), FN(exp(x) - 1));
    check(FN(cbrt(x)), FN(pow(x, 1.0 / 3.0)));
    check(FN(erfc(x)), FN(1 - erf(x)));
    check(FN(erf(tan(x))), FN(1 - erfc(tan(x))));
    check(FN(hypot(x, 2.0 * x)), FN(sqrt(5.0 * x * x)));
    check(FN(hypot(x, 2.0 * x, 3.0 * x)), FN(sqrt(14.0 * x * x)));
    check(FN(abs2(x)), FN(x * x));
    check(FN(conj(x)), FN(x));
    check(FN(fmod(5.0 * x, 0.4)), FN(5.0 * x - 1.2));
    check(FN(fmod(1.0, x)), FN(1.0 - 3.0 * x));
    check(FN(fmin(x, 0.2)), FN(0.2 + 0.0 * x));
    check(FN(fmax(x, 0.2)), FN(x));
    check(FN(copysign(x, -1.0)), FN(-x));
    check(FN(floor(10.0 * x) + ceil(x) + round(x) + trunc(x)), FN(4.0 + 0.0 * x));
    #undef FN

    // The value of fmod is exactly the one of std::fmod
    CHECK( val(fmod(dual(7.3), 2.1)) == std::fmod(7.3, 2.1) );
    CHECK( fmod(real(7.3), 2.1)[0] == std::fmod(7.3, 2.1) );
    CHECK( val(fmod(var(7.3), 2.1)) == std::fmod(7.3, 2.1) );

    // Second order derivatives of var
    var x = 0.3;
    const var fs[] = { log2(x), log1p(x), exp2(x), expm1(x), cbrt(x), erfc(x), fmod(5.0 * x, 0.4), fmod(1.0, x), copysign(x, -1.0) };
    const var gs[] = { log(x) / log(2.0), log(1 + x), exp(x * log(2.0)), exp(x) - 1, pow(x, 1.0 / 3.0), 1 - erf(x), 5.0 * x - 1.2, 1.0 - 3.0 * x, -x };
    for(auto i = 0; i < 9; ++i) {
        CHECK_APPROX( val(fs[i]), val(gs[i]) );
        const auto [dfdx] = derivativesx(fs[i], wrt(x));
        const auto [dgdx] = derivativesx(gs[i], wrt(x));
        CHECK_APPROX( val(dfdx), val(dgdx) );
        const auto [d2fdx2] = derivatives(dfdx, wrt(x));
        const auto [d2gdx2] = derivatives(dgdx, wrt(x));
        CHECK_APPROX( d2fdx2, d2gdx2 );
    }

    // Piecewise constant functions have zero derivatives, and expression trees of var are updated
    var u = 0.7;
    var h = log2(u) + cbrt(u) + exp2(u) + erfc(u) + floor(3 * u) + fmod(5 * u, 0.4);
    const auto [dhdu] = derivatives(h, wrt(u));
    CHECK_APPROX( dhdu, 1 / (0.7 * std::log(2.0)) + 1 / (3 * std::cbrt(0.49)) + std::log(2.0) * std::exp2(0.7) - 2 / std::sqrt(M_PI) * std::exp(-0.49) + 5.0 );
    u.update(0.9);
    h.update();
    CHECK_APPROX( val(h), std::log2(0.9) + std::cbrt(0.9) + std::exp2(0.9) + std::erfc(0.9) + std::floor(3 * 0.9) + std::fmod(5 * 0.9, 0.4) );
}

TEST_CASE("testing inverse and solve of Eigen matrices of var (#141)", "[issues][var][eigen]")
{
    // Eigen 3.4 triangular solvers use `cond ? Scalar : Scalar/Scalar`, ambiguous with two way var/ExprPtr conversions
    Eigen::Matrix<var, 6, 6> A = Eigen::Matrix<var, 6, 6>::Identity() * 2.0;
    A(0, 1) = 0.5;
    A(1, 0) = 0.5;
    var s = 3.0;
    A(5, 5) = s;
    const Eigen::Matrix<var, 6, 6> Ai = A.inverse();
    const Eigen::Matrix<var, 6, 1> b = Eigen::Matrix<var, 6, 1>::Ones();
    const Eigen::Matrix<var, 6, 1> x = A.lu().solve(b);
    var r = x.sum() + Ai.trace();
    const auto [drds] = derivatives(r, wrt(s));
    CHECK_APPROX( val(Ai(5, 5)), 1.0 / 3.0 );
    CHECK_APPROX( drds, -2.0 / 9.0 );

    var t = 2.0;
    const var e = false ? t : t * 2.0; // var and ExprPtr operands
    CHECK_APPROX( val(e), 4.0 );

    // Compound assignments with var operands, previously relying on the implicit conversion
    var u = 2.0;
    u += t; u -= t; u *= t; u /= t;
    CHECK_APPROX( val(u), 2.0 );
}
