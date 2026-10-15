// Device tests (NVRTC) of forward mode automatic differentiation with real numbers (Taylor coefficients).

#include <nvrtc_test.hpp>

#include <autodiff/forward/real.hpp>

using namespace autodiff;

namespace {

// Each thread evaluates the functions at a different point in (0.1, 0.42)
__device__ double point() { return 0.1 + 0.01 * threadIdx.x; }

constexpr double pi = 3.141592653589793;

} // namespace

AUTODIFF_NVRTC_TEST(real_arithmetic)
{
    const double a = point();
    real4th x = a;
    real4th y = 2.0;
    auto f = [](real4th x, real4th y) -> real4th { return (x + 1) * (x - y) / (2 * y) + x * x * x - 3 / x; };
    const double fx = (2 * a + 1 - 2.0) / (2 * 2.0) + 3 * a * a + 3 / (a * a);
    CHECK_APPROX(derivatives(f, along(1.0, 0.0), at(x, y))[1], fx);

    real1st u = a;
    CHECK_APPROX(derivative([](real1st u) -> real1st { return u * u; }, wrt(u), at(u)), 2 * a);

    real4th z = x;
    z += y; z -= 1.0; z *= x; z /= y;
    CHECK_APPROX(val(z), (a + 2.0 - 1.0) * a / 2.0);
    CHECK(x < y && y > x && x <= x && x != y && x == a);
}

AUTODIFF_NVRTC_TEST(real_functions)
{
    const double a = point();
    real4th x = a;
    using namespace autodiff::detail;
    // first order derivative along x
    auto d1 = [&](auto f) { return derivatives(f, along(1.0), at(x))[1]; };

    // (sin x)^(n) = sin(x + n pi/2)
    const auto ds = derivatives([](real4th x) -> real4th { return sin(x); }, along(1.0), at(x));
    for(int n = 0; n <= 4; ++n)
        CHECK_APPROX(ds[n], ::sin(a + n * pi / 2));

    const auto de = derivatives([](real4th x) -> real4th { return exp(2 * x); }, along(1.0), at(x));
    for(int n = 0; n <= 4; ++n)
        CHECK_APPROX(de[n], ::pow(2.0, n) * ::exp(2 * a));

    // (log x)^(n) = (-1)^(n-1) (n-1)! / x^n
    const auto dl = derivatives([](real4th x) -> real4th { return log(x); }, along(1.0), at(x));
    CHECK_APPROX(dl[1], 1 / a);
    CHECK_APPROX(dl[2], -1 / (a * a));
    CHECK_APPROX(dl[3], 2 / (a * a * a));
    CHECK_APPROX_TOL(dl[4], -6 / (a * a * a * a), 1e-11);

    CHECK_APPROX(d1([](real4th x) -> real4th { return cos(x); }), -::sin(a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return tan(x); }), 1 / (::cos(a) * ::cos(a)));
    CHECK_APPROX(d1([](real4th x) -> real4th { return asin(x); }), 1 / ::sqrt(1 - a * a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return acos(x); }), -1 / ::sqrt(1 - a * a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return atan(x); }), 1 / (1 + a * a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return sinh(x); }), ::cosh(a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return cosh(x); }), ::sinh(a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return tanh(x); }), 1 / (::cosh(a) * ::cosh(a)));
    CHECK_APPROX(d1([](real4th x) -> real4th { return log10(x); }), 1 / (a * ::log(10.0)));
    CHECK_APPROX(d1([](real4th x) -> real4th { return sqrt(x); }), 0.5 / ::sqrt(a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return cbrt(x); }), 1 / (3 * ::cbrt(a * a)));
    CHECK_APPROX(d1([](real4th x) -> real4th { return pow(x, 3.0); }), 3 * a * a);
    CHECK_APPROX(d1([](real4th x) -> real4th { return pow(2.0, x); }), ::log(2.0) * ::pow(2.0, a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return abs(-x); }), 1.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return sign(x) * x; }), 1.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return copysign(x, -1.0); }), -1.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return copysign(-x, -1.0); }), -1.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return copysign(-x, 1.0); }), 1.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return min(x, 2 * x); }), 1.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return max(x, 2 * x); }), 2.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return log2(x); }), 1 / (a * ::log(2.0)));
    CHECK_APPROX(d1([](real4th x) -> real4th { return log1p(x); }), 1 / (1 + a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return exp2(x); }), ::log(2.0) * ::exp2(a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return expm1(x); }), ::exp(a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return erf(x); }), 2 / ::sqrt(pi) * ::exp(-a * a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return erfc(x); }), -2 / ::sqrt(pi) * ::exp(-a * a));
    CHECK_APPROX(d1([](real4th x) -> real4th { return hypot(x, 2.0); }), a / ::hypot(a, 2.0));
    CHECK_APPROX(d1([](real4th x) -> real4th { return hypot(x, 2.0, 3.0); }), a / ::sqrt(a * a + 13.0));
    CHECK_APPROX(d1([](real4th x) -> real4th { return abs2(x); }), 2 * a);
    CHECK_APPROX(d1([](real4th x) -> real4th { return conj(x); }), 1.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return floor(x) + ceil(x) + round(x) + trunc(x); }), 0.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return fmin(x, 2 * x); }), 1.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return fmax(x, 2 * x); }), 2.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return fmod(3 * x, 0.07); }), 3.0);
    CHECK_APPROX(d1([](real4th x) -> real4th { return fmod(1.0, x); }), -::round((1.0 - ::fmod(1.0, a)) / a));
    // (log1p x)^(n) = (-1)^(n-1) (n-1)! / (1 + x)^n
    const auto dl1p = derivatives([](real4th x) -> real4th { return log1p(x); }, along(1.0), at(x));
    CHECK_APPROX(dl1p[0], ::log1p(a));
    CHECK_APPROX(dl1p[2], -1 / ((1 + a) * (1 + a)));
    CHECK_APPROX(dl1p[3], 2 / ((1 + a) * (1 + a) * (1 + a)));

    real4th y = 2.0 * a;
    CHECK_APPROX(derivatives([](real4th x, real4th y) -> real4th { return atan2(y, x); }, along(1.0, 0.0), at(x, y))[1], -2 * a / (5 * a * a));
}

AUTODIFF_NVRTC_TEST(real_taylorseries)
{
    const double a = point();
    real4th x = a;
    auto ts = taylorseries([](real4th x) -> real4th { return exp(x); }, along(1.0), at(x));
    const double t = 0.01;
    CHECK_APPROX_TOL(ts(t), ::exp(a + t), 1e-11);

    using Vec2 = stdx::array<real4th, 2>;
    Vec2 v = {a, 2 * a};
    const stdx::array<double, 2> dir = {1.0, -1.0};
    auto g = [](const Vec2& v) -> real4th { return v[0] * v[1]; };
    // g(v + t dir) = (a + t)(2a - t) = 2a^2 + a t - t^2
    const auto d = derivatives(g, along(dir), at(v));
    CHECK_APPROX(d[0], 2 * a * a);
    CHECK_APPROX(d[1], a);
    CHECK_APPROX(d[2], -2.0);
    CHECK_APPROX(d[3], 0.0);
}
