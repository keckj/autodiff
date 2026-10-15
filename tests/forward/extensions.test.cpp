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

// Host tests of the features added for GPU support (also exercised on device by the NVRTC tests in tests/nvrtc).

// C++ includes
#include <array>

// autodiff includes
#include <autodiff/common/binomialcoefficient.hpp>
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <tests/utils/catch.hpp>
using namespace autodiff;

TEST_CASE("testing binomial coefficients", "[forward][binomial]")
{
    using autodiff::detail::binomialCoefficient;
    using autodiff::detail::BinomialCoefficient;
    for(size_t i = 0; i <= 50; ++i)
    {
        CHECK( binomialCoefficient(i, 0) == 1.0 );
        CHECK( binomialCoefficient(i, i) == 1.0 );
        for(size_t j = 1; j < i; ++j) // check Pascal's rule, exact in double precision
            CHECK( binomialCoefficient(i, j) == binomialCoefficient(i - 1, j - 1) + binomialCoefficient(i - 1, j) );
    }
    static_assert(BinomialCoefficient<4, 2> == 6.0);
    static_assert(BinomialCoefficient<50, 25> == 126410606437752.0);
}

TEST_CASE("testing sign and copysign functions", "[forward][dual][real]")
{
    dual x = 1.5;
    CHECK( val(sign(x)) == 1.0 );
    CHECK( val(sign(-x)) == -1.0 );
    CHECK( val(sign(x - 1.5)) == 0.0 );
    CHECK( derivative([](dual x) -> dual { return sign(x); }, wrt(x), at(x)) == 0.0 );
    CHECK( derivative([](dual x) -> dual { return sign(x) * x * x; }, wrt(x), at(x)) == 3.0 );
    CHECK( val(copysign(x, -2.0)) == -1.5 );
    CHECK( val(copysign(-x, 2.0)) == 1.5 );
    CHECK( val(copysign(-x, -2.0)) == -1.5 );
    dual mx = -1.5;
    CHECK( derivative([](dual x) -> dual { return copysign(x, -2.0); }, wrt(mx), at(mx)) == 1.0 );
    CHECK( derivative([](dual x) -> dual { return copysign(x, -2.0); }, wrt(x), at(x)) == -1.0 );
    CHECK( derivative([](dual x) -> dual { return copysign(x * x, x); }, wrt(x), at(x)) == 3.0 );

    real2nd r = -1.5;
    CHECK( sign(r)[0] == -1.0 );
    CHECK( sign(r)[1] == 0.0 );
    CHECK( copysign(r, 1.0)[0] == 1.5 );
    CHECK( copysign(r, -1.0)[0] == -1.5 );
    CHECK( copysign(r, -r)[0] == 1.5 );
    const auto d = derivatives([](real2nd r) -> real2nd { return copysign(r * r * r, -1.0); }, along(1.0), at(r));
    CHECK( d[0] == -1.5 * 1.5 * 1.5 ); // r^3 < 0 already has the sign of -1
    CHECK( d[1] == 3 * 1.5 * 1.5 );
    CHECK( d[2] == -6 * 1.5 );
    const auto e = derivatives([](real2nd r) -> real2nd { return copysign(r * r * r, 1.0); }, along(1.0), at(r));
    CHECK( e[0] == 1.5 * 1.5 * 1.5 ); // -r^3 near r = -1.5
    CHECK( e[1] == -3 * 1.5 * 1.5 );
    CHECK( e[2] == 6 * 1.5 );
}

TEST_CASE("testing std::array as vector type", "[forward][dual][real]")
{
    std::array<dual, 3> x = {1.0, 2.0, 3.0};
    auto f = [](const std::array<dual, 3>& x) -> std::array<dual, 3> { return {x[0] * x[1], x[1] * x[2], x[2]}; };
    const auto dfdx1 = derivative(f, wrt(x[1]), at(x));
    static_assert(std::is_same_v<std::decay_t<decltype(dfdx1)>, std::array<double, 3>>);
    CHECK( dfdx1[0] == 1.0 );
    CHECK( dfdx1[1] == 3.0 );
    CHECK( dfdx1[2] == 0.0 );

    std::array<real2nd, 2> y = {2.0, 3.0};
    const std::array<double, 2> dir = {1.0, 1.0};
    const auto d = derivatives([](const std::array<real2nd, 2>& y) -> real2nd { return y[0] * y[1]; }, along(dir), at(y));
    CHECK( d[0] == 6.0 );
    CHECK( d[1] == 5.0 );
    CHECK( d[2] == 2.0 );

    static_assert(autodiff::detail::VectorSize<std::array<dual, 3>> == 3);
    static_assert(autodiff::detail::VectorSize<std::vector<dual>> == autodiff::detail::DynamicSize);
    static_assert(autodiff::detail::VectorSize<Vector3dual> == 3);
    static_assert(autodiff::detail::VectorSize<VectorXdual> == autodiff::detail::DynamicSize);
}

TEST_CASE("testing static sizes of gradient, jacobian and hessian", "[forward][utils][gradient]")
{
    auto f = [](const Vector3dual& x) -> dual { return x[0] * x[1] * x[2]; };
    auto F = [](const Vector3dual& x) -> Vector2dual { return Vector2dual(x[0] * x[1], x[2]); };
    auto h = [](const Vector3dual2nd& x) -> dual2nd { return x[0] * x[1] * x[2]; };

    Vector3dual x(1.0, 2.0, 3.0);
    Vector3dual2nd x2(1.0, 2.0, 3.0);

    // Fixed size vectors in the wrt list produce fixed size results
    const auto g = gradient(f, wrt(x), at(x));
    static_assert(std::is_same_v<std::decay_t<decltype(g)>, Eigen::Vector3d>);
    CHECK( g == Eigen::Vector3d(6.0, 3.0, 2.0) );

    const auto J = jacobian(F, wrt(x), at(x));
    static_assert(std::is_same_v<std::decay_t<decltype(J)>, Eigen::Matrix<double, 2, 3>>);
    CHECK( J(0, 0) == 2.0 );
    CHECK( J(0, 1) == 1.0 );
    CHECK( J(1, 2) == 1.0 );

    const auto H = hessian(h, wrt(x2), at(x2));
    static_assert(std::is_same_v<std::decay_t<decltype(H)>, Eigen::Matrix3d>);
    CHECK( H(0, 1) == 3.0 );
    CHECK( H(1, 2) == 1.0 );
    CHECK( H(0, 0) == 0.0 );

    // Mixing fixed size vectors and numbers
    dual y = 4.0;
    auto fy = [](const Vector3dual& x, dual y) -> dual { return x.sum() * y; };
    const auto gy = gradient(fy, wrt(x, y), at(x, y));
    static_assert(std::is_same_v<std::decay_t<decltype(gy)>, Eigen::Vector4d>);
    CHECK( gy == Eigen::Vector4d(4.0, 4.0, 4.0, 6.0) );

    // Any dynamic vector in the wrt list produces dynamic results (as in previous versions)
    VectorXdual z = x;
    auto fz = [](const VectorXdual& x) -> dual { return x.prod(); };
    const auto gz = gradient(fz, wrt(z), at(z));
    static_assert(std::is_same_v<std::decay_t<decltype(gz)>, Eigen::VectorXd>);
    CHECK( gz == Eigen::Vector3d(6.0, 3.0, 2.0) );
}
