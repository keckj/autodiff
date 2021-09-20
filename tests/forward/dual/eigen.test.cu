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

// Catch includes
#include <catch2/catch.hpp>

// autodiff includes
#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
using namespace autodiff;

template<typename T>
auto approx(T&& expr) -> Approx
{
    const double zero = std::numeric_limits<double>::epsilon();
    return Approx(val(std::forward<T>(expr))).margin(zero);
}

TEST_CASE("testing autodiff::dual (with eigen)", "[forward][dual][eigen]")
{
    SECTION("testing array-unpacking of derivatives for eigen vector of dual numbers")
    {
        dual4th x;
        detail::seed<0>(x, 2.0);
        detail::seed<1>(x, 3.0);
        detail::seed<2>(x, 4.0);
        detail::seed<3>(x, 5.0);
        detail::seed<4>(x, 6.0);

        dual4th y;
        detail::seed<0>(y, 3.0);
        detail::seed<1>(y, 4.0);
        detail::seed<2>(y, 5.0);
        detail::seed<3>(y, 6.0);
        detail::seed<4>(y, 7.0);

        dual4th z;
        detail::seed<0>(z, 4.0);
        detail::seed<1>(z, 5.0);
        detail::seed<2>(z, 6.0);
        detail::seed<3>(z, 7.0);
        detail::seed<4>(z, 8.0);

        VectorX<dual4th,3> u(x, y, z);

        auto [u0, u1, u2, u3, u4] = derivatives(u);

        CHECK( u0[0] == approx(derivative<0>(x)) );
        CHECK( u0[1] == approx(derivative<0>(y)) );
        CHECK( u0[2] == approx(derivative<0>(z)) );

        CHECK( u1[0] == approx(derivative<1>(x)) );
        CHECK( u1[1] == approx(derivative<1>(y)) );
        CHECK( u1[2] == approx(derivative<1>(z)) );

        CHECK( u2[0] == approx(derivative<2>(x)) );
        CHECK( u2[1] == approx(derivative<2>(y)) );
        CHECK( u2[2] == approx(derivative<2>(z)) );

        CHECK( u3[0] == approx(derivative<3>(x)) );
        CHECK( u3[1] == approx(derivative<3>(y)) );
        CHECK( u3[2] == approx(derivative<3>(z)) );

        CHECK( u4[0] == approx(derivative<4>(x)) );
        CHECK( u4[1] == approx(derivative<4>(y)) );
        CHECK( u4[2] == approx(derivative<4>(z)) );
    }

    SECTION("testing casting to VectorXd")
    {
        VectorX<dual,3> x(0.5, 0.2, 0.3);

        VectorX<double,3> y = x.cast<double>();

        for(auto i = 0; i < 3; ++i)
            CHECK( x[i] == approx(y(i)) );
    }

    SECTION("testing casting to MatriXd")
    {
        MatrixX<dual,2,2> x;
        x << 0.5, 0.2, 0.3, 0.7;

        MatrixX<double,2,2> y = x.cast<double>();

        for(auto i = 0; i < 2; ++i)
            for(auto j = 0; j < 2; ++j)
                CHECK(x(i, j) == approx(y(i, j)));
    }

    SECTION("testing multiplication of VectorXdual by MatrixXd")
    {
        MatrixX<dual,2,2> A;
        A << 1.0, 3.0, 5.0, 7.0;

        VectorX<dual,2> x;

        detail::seed<0>(x[0], 1.0);
        detail::seed<0>(x[1], 2.0);

        detail::seed<1>(x[0], 3.0);
        detail::seed<1>(x[1], 5.0);

        VectorX<dual,2> b = A * x;

        CHECK( derivative<0>(b[0]) == approx(7.0) );
        CHECK( derivative<0>(b[1]) == approx(19.0) );

        CHECK( derivative<1>(b[0]) == approx(18.0) );
        CHECK( derivative<1>(b[1]) == approx(50.0) );
    }

    SECTION("testing class template specializations for Eigen::ScalarBinaryOpTraits")
    {
        dual4th x = 4.0;

        MatrixX<double,2,2> A;
        A << 1.0, 3.0, 5.0, 7.0;

        // The checks below also allow us to determine if compilation succeeds.
        // Note that we mix not only dual4th times MatrixXd, but also expressions,
        // such as UnaryExp -x and +x, BinaryExpr (x+x) and (x+x)*(x+x).

        CHECK( (x*A).isApprox(x*A.cast<dual4th>()) );
        CHECK( ((+x)*A).isApprox(+x*A.cast<dual4th>()) );
        CHECK( ((-x)*A).isApprox(-x*A.cast<dual4th>()) );
        CHECK( ((x+x)*A).isApprox(2*x*A.cast<dual4th>()) );
        CHECK( (((x+x)*(x+x))*A).isApprox(4*x*x*A.cast<dual4th>()) );

        CHECK( (A*x).isApprox(x*A.cast<dual4th>()) );
        CHECK( (A*(+x)).isApprox(+x*A.cast<dual4th>()) );
        CHECK( (A*(-x)).isApprox(-x*A.cast<dual4th>()) );
        CHECK( (A*(x+x)).isApprox(2*x*A.cast<dual4th>()) );
        CHECK( (A*((x+x)*(x+x))).isApprox(4*x*x*A.cast<dual4th>()) );
    }

    SECTION("using Eigen::VectorXdual")
    {
        SECTION("testing gradient derivatives")
        {
            auto f = [](const VectorX<dual,3>& x) -> dual
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorX<dual,3> x(1.0, 2.0, 3.0);

            VectorX<double,3> g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }

        SECTION("testing gradient derivatives of only the last two variables")
        {
            auto f = [](const VectorX<dual,3>& x) -> dual
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorX<dual,3> x(1.0, 2.0, 3.0);

            VectorX<double,2> g = gradient(f, wrt(x.tail<2>()), at(x));

            CHECK( g[0] == approx(x[1]) );
            CHECK( g[1] == approx(x[2]) );
        }

        SECTION("testing jacobian derivatives")
        {
            auto f = [](const VectorX<dual,3>& x) -> VectorX<dual,3>
            {
                return x / x.array().sum();
            };

            VectorX<dual,3> x(0.5, 0.2, 0.3);

            VectorX<dual,3> F;

            const MatrixX<double,3,3> J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }

        SECTION("testing jacobian derivatives of only the last two variables")
        {
            auto f = [](const VectorX<dual,3>& x) -> VectorX<dual,3>
            {
                return x / x.array().sum();
            };

            VectorX<dual,3> x(0.5, 0.2, 0.3);

            VectorX<dual,3> F;

            const MatrixX<double,3,2> J = jacobian(f, wrt(x.tail<2>()), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 2; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j + 1) ? 1.0 : 0.0)) );
        }

        SECTION("testing casting to VectorXd")
        {
            VectorX<dual,3> x(0.5, 0.2, 0.3);

            VectorX<double,3> y = x.cast<double>();

            for(auto i = 0; i < 3; ++i)
                CHECK( x(i) == approx(y(i)) );
        }

        SECTION("testing casting to VectorXf")
        {
            MatrixX<dual,2,2> x;
            x << 0.5, 0.2, 0.3, 0.7;
            MatrixX<double,2,2> y = x.cast<double>();
            for(auto i = 0; i < 2; ++i)
                for(auto j = 0; j < 2; ++j)
                    CHECK( x(i, j) == approx(y(i, j)) );
        }

        SECTION("test gradient size with respect to few arguments")
        {
            auto f = [](const VectorX<dual,3>& x, dual y, const VectorX<dual,4>& z) -> dual
            {
                return 0.5 * (( x.array() * x.array() ).sum() + y * y + (z.array() * z.array()).sum());
            };

            VectorX<dual,3> x(1.0, 2.0, 3.0);

            dual y = 2;

            VectorX<dual,4> z(1.0, 2.0, 3.0, 4.0);

            VectorX<double,7> g = gradient(f, wrt(x.tail<2>(), y, z), at(x, y, z));
            CHECK( g.size() == 7 );
        }

        SECTION("testing gradient derivatives wrt pack variables")
        {
            auto f = [](const VectorX<dual,2>& x, dual y, const VectorX<dual,1>& z) -> dual
            {
                return 0.5 * (( x.array() * x.array() ).sum() + y * y + (z.array() * z.array()).sum());
            };

            VectorX<dual,2> x(1.0, 2.0);

            dual y = 3.0;

            VectorX<dual,1> z(4.0);

            VectorX<double,4> g = gradient(f, wrt(x, y, z), at(x, y, z));

            CHECK(g[0] == approx(x[0]));
            CHECK(g[1] == approx(x[1]));
            CHECK(g[2] == approx(y));
            CHECK(g[3] == approx(z[0]));
        }

        SECTION("test jacobian size with respect to few arguments")
        {
            auto f = [](const VectorX<dual,3>& x, dual y, const VectorX<dual,3>& z) -> VectorX<dual,6>
            {
                VectorX<dual,6> ret;
                ret.head<3>() = x * y / x.array().sum();
                ret.tail<3>() = y * z;

                return ret;
            };

            VectorX<dual,3> x(0.5, 0.2, 0.3);

            dual y = 2.0;

            VectorX<dual,3> z(1.0, 2.0, 3.0);

            VectorX<dual,6> F;

            const MatrixX<double,6,3> J = jacobian(f, wrt(x.tail<2>(), y), at(x, y, z), F);

            CHECK(J.rows() == 6);
            CHECK(J.cols() == 3);
        }

        SECTION("test jacobian size with respect to few arguments")
        {
            auto f = [](const VectorX<dual,3>& x, dual y, const VectorX<dual,3>& z) -> VectorX<dual,6>
            {
                VectorX<dual,6> ret;
                ret.head<3>() = x * y / x.array().sum();
                ret.tail<3>() = y * z;

                return ret;
            };

            VectorX<dual,3> x(0.5, 0.2, 0.3);

            dual y = 2.0;

            VectorX<dual,3> z(1.0, 2.0, 3.0);

            VectorX<dual,6> F;

            const MatrixX<double,6,7> J = jacobian(f, wrt(x, y, z), at(x, y, z), F);

            for (auto i = 0; i < 3; ++i)
                for (auto j = 0; j < 3; ++j)
                    CHECK(J(i, j) == approx(-F[i] + ((i == j) ? y.val : 0.0)));

            for (auto i = 0; i < 3; ++i)
                for (auto j = 0; j < 3; ++j)
                    CHECK(J(i + 3, j) == approx(0.0));

            for (auto i = 0; i < 6; ++i)
                    CHECK(J(i, 3) == approx( i < 3 ? x(i) : z(i - 3)));

            for (auto i = 0; i < 3; ++i)
                for (auto j = 0; j < 3; ++j)
                    CHECK(J(i, j + 4) == approx(0.0));

            for (auto i = 0; i < 3; ++i)
                for (auto j = 0; j < 3; ++j)
                    CHECK(J(i + 3, j + 4) == approx((i == j) ? y.val : 0.0));
        }
    }

    SECTION("using VectorXdual2nd")
    {
        SECTION("testing casting to VectorXd")
        {
            VectorX<dual2nd,3> x(1.0, 2.0, 3.0);

            VectorX<double,3> y = x.template cast<double>();

            for(auto i = 0; i < 3; ++i)
                CHECK( x(i) == approx(y(i)) );
        }

        using dual2nd = HigherOrderDual<2, double>;

        SECTION("testing gradient derivatives")
        {
            auto f = [](const VectorX<dual2nd,3>& x) -> dual2nd
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorX<dual2nd,3> x(1.0, 2.0, 3.0);

            VectorX<double,3> g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }

        SECTION("testing jacobian derivatives")
        {
            auto f = [](const VectorX<dual2nd,3>& x) -> VectorX<dual2nd,3>
            {
                return x / x.array().sum();
            };

            VectorX<dual2nd,3> x(0.5, 0.2, 0.3);

            VectorX<dual2nd,3> F;

            const MatrixX<double,3,3> J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }

        SECTION("testing hessian derivatives")
        {
            auto f = [](const VectorX<dual2nd,3>& x) -> dual2nd
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorX<dual2nd,3> x(1.0, 2.0, 3.0);

            MatrixX<double,3,3> H = hessian(f, wrt(x), at(x));

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( H(i, j) == approx(((i == j) ? 1.0 : 0.0)) );
        }
    }

    SECTION("using VectorXdual3rd")
    {
        using dual3rd = HigherOrderDual<2, double>;

        SECTION("testing casting to VectorXd")
        {
            VectorX<dual3rd,3> x(1.0, 2.0, 3.0);

            VectorX<double,3> y = x.template cast<double>();

            for(auto i = 0; i < 3; ++i)
                CHECK( x(i) == approx(y(i)) );
        }

        SECTION("testing gradient derivatives")
        {
            auto f = [](const VectorX<dual3rd,3>& x) -> dual3rd
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorX<dual3rd,3> x(1.0, 2.0, 3.0);

            VectorX<double,3> g = gradient(f, wrt(x), at(x));

            CHECK( g[0] == approx(x[0]) );
            CHECK( g[1] == approx(x[1]) );
            CHECK( g[2] == approx(x[2]) );
        }

        SECTION("testing jacobian derivatives")
        {
            auto f = [](const VectorX<dual3rd,3>& x) -> VectorX<dual3rd,3>
            {
                return x / x.array().sum();
            };

            VectorX<dual3rd,3> x(0.5, 0.2, 0.3);

            VectorX<dual3rd,3> F;

            const MatrixX<double,3,3> J = jacobian(f, wrt(x), at(x), F);

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( J(i, j) == approx(-F[i] + ((i == j) ? 1.0 : 0.0)) );
        }

        SECTION("testing hessian derivatives")
        {
            auto f = [](const VectorX<dual3rd,3>& x) -> dual3rd
            {
                return 0.5 * ( x.array() * x.array() ).sum();
            };

            VectorX<dual3rd,3> x(1.0, 2.0, 3.0);

            MatrixX<double,3,3> H = hessian(f, wrt(x), at(x));

            for(auto i = 0; i < 3; ++i)
                for(auto j = 0; j < 3; ++j)
                    CHECK( H(i, j) == approx(((i == j) ? 1.0 : 0.0)) );
        }
    }
}
