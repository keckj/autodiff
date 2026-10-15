// Checks the standard library abstraction layer of autodiff under NVRTC.
#include <nvrtc_test.hpp>
#include <autodiff/common/stdlib.hpp>

using namespace autodiff;

AUTODIFF_NVRTC_TEST(stdlib_math)
{
    double x = 0.5;
    float y = 0.5f;
    CHECK_APPROX(stdmath::sin(x) * stdmath::sin(x) + stdmath::cos(x) * stdmath::cos(x), 1.0);
    CHECK_APPROX_TOL(stdmath::sin(y) * stdmath::sin(y) + stdmath::cos(y) * stdmath::cos(y), 1.0, 1e-6);
    CHECK_APPROX(stdmath::abs(-x), x);
    CHECK_APPROX(stdmath::abs(-y), y);
    CHECK_APPROX(stdmath::pow(x, 2.0), 0.25);
    CHECK_APPROX(stdmath::hypot(3.0, 4.0), 5.0);
    CHECK_APPROX(stdmath::hypot3(2.0, 3.0, 6.0), 7.0);
    CHECK_APPROX(stdmath::cbrt(27.0), 3.0);
    CHECK_APPROX(stdmath::copysign(2.0, -1.0), -2.0);
    CHECK_APPROX(stdmath::max(1.0, 2.0), 2.0);
    CHECK_APPROX(stdmath::min(1.0, 2.0), 1.0);
    CHECK_APPROX(stdmath::atanh(stdmath::tanh(x)), x);
    CHECK_APPROX(stdmath::asinh(stdmath::sinh(x)), x);
    CHECK_APPROX(stdmath::acosh(stdmath::cosh(2.0)), 2.0);
    CHECK_APPROX(stdmath::log10(stdmath::pow(10.0, x)), x);
    CHECK_APPROX(stdmath::atan2(1.0, 1.0) * 4.0, 3.141592653589793);
    CHECK_APPROX_TOL(stdmath::erf(0.5), 0.5204998778130465, 1e-15);
    CHECK_APPROX_TOL(stdmath::erfc(0.5), 1 - 0.5204998778130465, 1e-15);
    CHECK_APPROX(stdmath::exp2(3.0), 8.0);
    CHECK_APPROX(stdmath::log2(8.0), 3.0);
    CHECK_APPROX(stdmath::log1p(stdmath::expm1(x)), x);
    CHECK(stdmath::floor(-1.5) == -2.0 && stdmath::ceil(-1.5) == -1.0);
    CHECK(stdmath::round(-1.5) == -2.0 && stdmath::trunc(-1.5) == -1.0);
    CHECK(stdmath::floor(-1.5f) == -2.0f && stdmath::trunc(-1.5f) == -1.0f);
    CHECK(stdmath::fmod(7.0, 3.0) == 1.0);
    CHECK(stdmath::fmin(1.0, 2.0) == 1.0 && stdmath::fmax(1.0, 2.0) == 2.0);
}

AUTODIFF_NVRTC_TEST(stdlib_utilities)
{
    stdx::array<double, 3> a = {1.0, 2.0, 3.0};
    CHECK(a.size() == 3);
    auto t = stdx::make_tuple(1, 2.0, a);
    CHECK(stdx::get<0>(t) == 1);
    CHECK(stdx::get<2>(t)[2] == 3.0);
    static_assert(stdx::is_same_v<stdx::tuple_element_t<1, decltype(t)>, double>);
    CHECK(stdx::apply([](int i, double d, const stdx::array<double, 3>&) { return i + d; }, t) == 3.0);
}
