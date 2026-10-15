// Device tests (NVRTC) of forward mode automatic differentiation with dual numbers.

#include <nvrtc_test.hpp>

#include <autodiff/forward/dual.hpp>

using namespace autodiff;

namespace {

// Each thread evaluates the functions at a different point in (0.1, 0.42)
__device__ double point() { return 0.1 + 0.01 * threadIdx.x; }

} // namespace

AUTODIFF_NVRTC_TEST(dual_arithmetic)
{
    const double a = point();
    dual x = a;
    dual y = 2.0;

    auto f = [](dual x, dual y) -> dual { return (x + 1) * (x - y) / (2 * y) + x * x * x - 3 / x; };
    const double fx = (2 * a + 1 - 2.0) / (2 * 2.0) + 3 * a * a + 3 / (a * a);
    const double fy = (a + 1) * (-2.0 * 2.0 - 2 * (a - 2.0)) / (4 * 2.0 * 2.0);
    CHECK_APPROX(derivative(f, wrt(x), at(x, y)), fx);
    CHECK_APPROX(derivative(f, wrt(y), at(x, y)), fy);

    dual z = x;
    z += y; z -= 1.0; z *= x; z /= y;
    CHECK_APPROX(val(z), (a + 2.0 - 1.0) * a / 2.0);
    CHECK(x < y && y > x && x <= x && x != y && x == a);
    CHECK(-x == -a);
}

AUTODIFF_NVRTC_TEST(dual_functions)
{
    const double a = point();
    dual x = a;
    using namespace autodiff::detail;

    CHECK_APPROX(derivative([](dual x) -> dual { return sin(x); }, wrt(x), at(x)), ::cos(a));
    CHECK_APPROX(derivative([](dual x) -> dual { return cos(x); }, wrt(x), at(x)), -::sin(a));
    CHECK_APPROX(derivative([](dual x) -> dual { return tan(x); }, wrt(x), at(x)), 1 / (::cos(a) * ::cos(a)));
    CHECK_APPROX(derivative([](dual x) -> dual { return asin(x); }, wrt(x), at(x)), 1 / ::sqrt(1 - a * a));
    CHECK_APPROX(derivative([](dual x) -> dual { return acos(x); }, wrt(x), at(x)), -1 / ::sqrt(1 - a * a));
    CHECK_APPROX(derivative([](dual x) -> dual { return atan(x); }, wrt(x), at(x)), 1 / (1 + a * a));
    CHECK_APPROX(derivative([](dual x) -> dual { return sinh(x); }, wrt(x), at(x)), ::cosh(a));
    CHECK_APPROX(derivative([](dual x) -> dual { return cosh(x); }, wrt(x), at(x)), ::sinh(a));
    CHECK_APPROX(derivative([](dual x) -> dual { return tanh(x); }, wrt(x), at(x)), 1 / (::cosh(a) * ::cosh(a)));
    CHECK_APPROX(derivative([](dual x) -> dual { return asinh(x); }, wrt(x), at(x)), 1 / ::sqrt(a * a + 1));
    CHECK_APPROX(derivative([](dual x) -> dual { return acosh(x + 1); }, wrt(x), at(x)), 1 / ::sqrt((a + 1) * (a + 1) - 1));
    CHECK_APPROX(derivative([](dual x) -> dual { return atanh(x); }, wrt(x), at(x)), 1 / (1 - a * a));
    CHECK_APPROX(derivative([](dual x) -> dual { return exp(x); }, wrt(x), at(x)), ::exp(a));
    CHECK_APPROX(derivative([](dual x) -> dual { return log(x); }, wrt(x), at(x)), 1 / a);
    CHECK_APPROX(derivative([](dual x) -> dual { return log10(x); }, wrt(x), at(x)), 1 / (a * ::log(10.0)));
    CHECK_APPROX(derivative([](dual x) -> dual { return sqrt(x); }, wrt(x), at(x)), 0.5 / ::sqrt(a));
    CHECK_APPROX(derivative([](dual x) -> dual { return pow(x, 3.0); }, wrt(x), at(x)), 3 * a * a);
    CHECK_APPROX(derivative([](dual x) -> dual { return pow(2.0, x); }, wrt(x), at(x)), ::log(2.0) * ::pow(2.0, a));
    CHECK_APPROX(derivative([](dual x) -> dual { return pow(x, x); }, wrt(x), at(x)), ::pow(a, a) * (::log(a) + 1));
    CHECK_APPROX(derivative([](dual x) -> dual { return abs(-x); }, wrt(x), at(x)), 1.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return erf(x); }, wrt(x), at(x)), 2 / ::sqrt(3.141592653589793) * ::exp(-a * a));
    CHECK_APPROX(derivative([](dual x) -> dual { return sign(x) * x; }, wrt(x), at(x)), 1.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return copysign(x, -1.0); }, wrt(x), at(x)), -1.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return copysign(-x, -1.0); }, wrt(x), at(x)), -1.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return copysign(-x, 1.0); }, wrt(x), at(x)), 1.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return min(x, 2 * x); }, wrt(x), at(x)), 1.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return max(x, 2 * x); }, wrt(x), at(x)), 2.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return log2(x); }, wrt(x), at(x)), 1 / (a * ::log(2.0)));
    CHECK_APPROX(derivative([](dual x) -> dual { return log1p(x); }, wrt(x), at(x)), 1 / (1 + a));
    CHECK_APPROX(derivative([](dual x) -> dual { return exp2(x); }, wrt(x), at(x)), ::log(2.0) * ::exp2(a));
    CHECK_APPROX(derivative([](dual x) -> dual { return expm1(x); }, wrt(x), at(x)), ::exp(a));
    CHECK_APPROX(derivative([](dual x) -> dual { return cbrt(x); }, wrt(x), at(x)), 1 / (3 * ::cbrt(a * a)));
    CHECK_APPROX(derivative([](dual x) -> dual { return erfc(x); }, wrt(x), at(x)), -2 / ::sqrt(3.141592653589793) * ::exp(-a * a));
    CHECK_APPROX(derivative([](dual x) -> dual { return floor(x) + ceil(x) + round(x) + trunc(x); }, wrt(x), at(x)), 0.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return fmin(x, 2 * x); }, wrt(x), at(x)), 1.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return fmax(x, 2 * x); }, wrt(x), at(x)), 2.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return fmod(3 * x, 0.07); }, wrt(x), at(x)), 3.0);
    CHECK_APPROX(derivative([](dual x) -> dual { return fmod(1.0, x); }, wrt(x), at(x)), -::round((1.0 - ::fmod(1.0, a)) / a));
    CHECK(val(fmod(x, 0.07)) == ::fmod(a, 0.07));

    dual y = 2.0 * a;
    dual z = 3.0 * a;
    CHECK_APPROX(derivative([](dual x, dual y) -> dual { return atan2(y, x); }, wrt(x), at(x, y)), -2 * a / (5 * a * a));
    CHECK_APPROX(derivative([](dual x, dual y) -> dual { return hypot(x, y); }, wrt(x), at(x, y)), 1 / ::sqrt(5.0));
    CHECK_APPROX(derivative([](dual x, dual y, dual z) -> dual { return hypot(x, y, z); }, wrt(x), at(x, y, z)), 1 / ::sqrt(14.0));
    CHECK_APPROX(derivative([](dual x, dual y, dual z) -> dual { return hypot(x, y, z); }, wrt(z), at(x, y, z)), 3 / ::sqrt(14.0));
}

AUTODIFF_NVRTC_TEST(dual_higher_order)
{
    const double a = point();
    dual4th x = a;
    auto f = [](dual4th x) -> dual4th { return sin(x) * exp(x); };
    const auto d = derivatives(f, wrt(x), at(x));
    static_assert(stdx::tuple_size_v<decltype(d)> == 5);
    // (sin x e^x)^(n) = 2^(n/2) e^x sin(x + n pi/4)
    for(int n = 0; n <= 4; ++n)
        CHECK_APPROX(d[n], ::pow(2.0, 0.5 * n) * ::exp(a) * ::sin(a + n * 3.141592653589793 / 4));

    dual2nd u = a, v = 2 * a;
    auto g = [](dual2nd u, dual2nd v) -> dual2nd { return u * u * v; };
    CHECK_APPROX((derivative<2>(g, wrt(u, v), at(u, v))), 2 * a);
    CHECK_APPROX((derivative<2>(g, wrt(u, u), at(u, v))), 2 * 2 * a);
}

AUTODIFF_NVRTC_TEST(dual_arrays)
{
    using Vec3 = stdx::array<dual, 3>;
    const double a = point();
    Vec3 x = {a, 2 * a, 3 * a};
    auto f = [](const Vec3& x) -> dual { return x[0] * x[1] + sin(x[2]); };
    // directional derivative along (1, 1, 1)
    const stdx::array<double, 3> dir = {1.0, 1.0, 1.0};
    const double expected = 2 * a + a + ::cos(3 * a);
    CHECK_APPROX(derivatives(f, along(dir), at(x))[1], expected);

    auto g = [](const Vec3& x) -> Vec3 { return Vec3{x[0] * x[0], 2 * x[1], x[2]}; };
    const auto dg = derivative(g, wrt(x[0]), at(x));
    static_assert(stdx::is_same_v<stdx::remove_cv_t<decltype(dg)>, stdx::array<double, 3>>);
    CHECK_APPROX(dg[0], 2 * a);
    CHECK_APPROX(dg[1], 0.0);
}
