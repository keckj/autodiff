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

// Catch includes
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// autodiff includes
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

// Expression trees of var objects created in device code live on the device heap and can not be transferred to the
// host: each test builds its variables in a kernel and only copies back plain numbers.

namespace {

template<typename Result, typename Callable>
__global__ void deviceKernel(Result* result, Callable callee) { *result = callee(); }

/// Evaluate callee() in a single device thread and return its (trivially copyable) result. The result type must be
/// given explicitly since the return type of an extended __device__ lambda can not be queried in host code.
template<typename Result, typename Callable>
auto onDevice(Callable&& callee) -> Result
{
    Result result{};
    Result* resultDevicePtr;
    CHECK(cudaMalloc(&resultDevicePtr, sizeof(Result)) == cudaSuccess);
    deviceKernel<<<1, 1>>>(resultDevicePtr, callee);
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaMemcpy(&result, resultDevicePtr, sizeof(Result), cudaMemcpyDeviceToHost) == cudaSuccess);
    CHECK(cudaFree(resultDevicePtr) == cudaSuccess);
    return result;
}

__device__ double grad(const var& y, var& x) { return derivatives(y, wrt(x))[0]; }

using Array2 = stdx::array<double, 2>;
using Array4 = stdx::array<double, 4>;
using Array6 = stdx::array<double, 6>;
using Array12 = stdx::array<double, 12>;

} // namespace

TEST_CASE("testing autodiff::var on device", "[reverse][var][cuda]")
{
    SECTION("default memory management on device")
    {
        static_assert(!AUTODIFF_VAR_USE_STD_MEMORY, "var must not use std::shared_ptr/std::function with a CUDA compiler");
    }

    SECTION("arithmetic operators")
    {
        const auto r = onDevice<Array4>([] __device__ () {
            var x = 0.5;
            var y = 2.0;
            var f = (x + 1) * (x - y) / (2 * y) + x * x * x - 3 / x;
            const auto [fx, fy] = derivatives(f, wrt(x, y));
            return Array4{val(f), fx, fy, grad(-x, x)};
        });
        const double a = 0.5;
        CHECK( r[0] == Catch::Approx((a + 1) * (a - 2.0) / 4.0 + a * a * a - 3 / a) );
        CHECK( r[1] == Catch::Approx((2 * a + 1 - 2.0) / 4.0 + 3 * a * a + 3 / (a * a)) );
        CHECK( r[2] == Catch::Approx((a + 1) * (-4.0 - 2 * (a - 2.0)) / 16.0) );
        CHECK( r[3] == -1.0 );
    }

    SECTION("mathematical functions")
    {
        const auto r = onDevice<Array12>([] __device__ () {
            var x = 0.3;
            var y = 1.1;
            Array12 d;
            d[0] = grad(sin(x), x);
            d[1] = grad(cos(x), x);
            d[2] = grad(tan(x), x);
            d[3] = grad(exp(x), x);
            d[4] = grad(log(x), x);
            d[5] = grad(sqrt(x), x);
            d[6] = grad(pow(x, y), y);
            d[7] = grad(atan2(x, y), x);
            d[8] = grad(hypot(x, y), x);
            d[9] = grad(hypot(x, y, 2.0), x);
            d[10] = grad(erf(x), x);
            d[11] = grad(abs(-x), x);
            return d;
        });
        const double x = 0.3, y = 1.1;
        CHECK( r[0] == Catch::Approx(std::cos(x)) );
        CHECK( r[1] == Catch::Approx(-std::sin(x)) );
        CHECK( r[2] == Catch::Approx(1 / (std::cos(x) * std::cos(x))) );
        CHECK( r[3] == Catch::Approx(std::exp(x)) );
        CHECK( r[4] == Catch::Approx(1 / x) );
        CHECK( r[5] == Catch::Approx(0.5 / std::sqrt(x)) );
        CHECK( r[6] == Catch::Approx(std::pow(x, y) * std::log(x)) );
        CHECK( r[7] == Catch::Approx(y / (x * x + y * y)) );
        CHECK( r[8] == Catch::Approx(x / std::hypot(x, y)) );
        CHECK( r[9] == Catch::Approx(x / std::sqrt(x * x + y * y + 4.0)) );
        CHECK( r[10] == Catch::Approx(2 / std::sqrt(3.14159265358979323846) * std::exp(-x * x)) );
        CHECK( r[11] == 1.0 );
    }

    SECTION("conditions and updates")
    {
        const auto r = onDevice<Array6>([] __device__ () {
            var x = 0.2;
            var y = 0.25;
            var c = condition(x < y, x * x, 3 * x);
            var n = condition(!(x < y), x * x, 3 * x);
            Array6 d;
            d[0] = val(c);
            d[1] = val(n);
            x.update(1.0);
            c.update();
            n.update();
            d[2] = val(c);
            d[3] = grad(c, x);
            d[4] = val(n);
            d[5] = val(max(x, y));
            return d;
        });
        CHECK( r[0] == Catch::Approx(0.04) );
        CHECK( r[1] == Catch::Approx(0.6) );
        CHECK( r[2] == Catch::Approx(3.0) );
        CHECK( r[3] == Catch::Approx(3.0) );
        CHECK( r[4] == Catch::Approx(1.0) );
        CHECK( r[5] == Catch::Approx(1.0) );
    }

    SECTION("higher order derivatives")
    {
        const auto r = onDevice<Array4>([] __device__ () {
            var x = 0.7;
            var y = 2.0;
            var f = x * x * y + sin(x);
            const auto [fx, fy] = derivativesx(f, wrt(x, y));
            const auto [fxx, fxy] = derivatives(fx, wrt(x, y));
            return Array4{val(fx), val(fy), fxx, fxy};
        });
        CHECK( r[0] == Catch::Approx(2 * 0.7 * 2.0 + std::cos(0.7)) );
        CHECK( r[1] == Catch::Approx(0.7 * 0.7) );
        CHECK( r[2] == Catch::Approx(4.0 - std::sin(0.7)) );
        CHECK( r[3] == Catch::Approx(2 * 0.7) );
    }

    SECTION("gradient and hessian with eigen")
    {
        const auto r = onDevice<Array6>([] __device__ () {
            Vector3var x(0.5, 2.0, 3.0);
            var y = x[0] * x[1] * x[2] + x.squaredNorm();
            Eigen::Vector3d g = gradient(y, x);
            Eigen::Vector3d gx;
            Eigen::Matrix3d H = hessian(y, x, gx);
            Array6 d = {g[0], g[1], g[2], H(0, 1), H(1, 2), H(2, 2)};
            return d;
        });
        CHECK( r[0] == Catch::Approx(6.0 + 1.0) );
        CHECK( r[1] == Catch::Approx(1.5 + 4.0) );
        CHECK( r[2] == Catch::Approx(1.0 + 6.0) );
        CHECK( r[3] == Catch::Approx(3.0) );
        CHECK( r[4] == Catch::Approx(0.5) );
        CHECK( r[5] == Catch::Approx(2.0) );
    }

    SECTION("expression trees are released on device")
    {
        const auto r = onDevice<Array2>([] __device__ () {
            var x = 0.5;
            Array2 counts;
            {
                var y = sin(x) * x + exp(x) - x * x;
                counts[0] = double(x.expr.use_count() > 1);
            }
            counts[1] = double(x.expr.use_count());
            return counts;
        });
        CHECK( r[0] == 1.0 );
        CHECK( r[1] == 1.0 );
    }
}
