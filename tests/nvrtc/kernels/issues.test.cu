// Device tests (NVRTC) of bugs reported upstream (https://github.com/autodiff/autodiff/issues). The forward and reverse
// modes are deliberately included together (issue #179).

// Eigen must be included first, before any header using std:: (see ext/eigen-nvrtc/NVRTC.md)
#include <Eigen/Core>

#include <nvrtc_test.hpp>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
#include <autodiff/forward/utils/taylorseries.hpp>
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using namespace autodiff;

namespace {

// Each thread evaluates the functions at a different point in (0.1, 0.42)
__device__ double point() { return 0.1 + 0.01 * threadIdx.x; }

template<typename X>
__device__ auto fsinx(const X& x) -> X { return sin(x) * x; }

} // namespace

AUTODIFF_NVRTC_TEST(issues_forward_reverse_together)
{
    const double a = point();
    dual x = a;
    CHECK_APPROX(derivative([](dual x) -> dual { return x * x; }, wrt(x), at(x)), 2 * a);

    var v = a;
    var y = v * v;
    CHECK_APPROX(derivatives(y, wrt(v))[0], 2 * a);

    real r = a;
    const real cr = 2 * a;
    CHECK(val(r) == a && val(cr) == 2 * a && val(x) == a && val(v) == a);
    CHECK(eval(r)[0] == a);
}

AUTODIFF_NVRTC_TEST(issues_real_higher_order_wrt)
{
    // Kernel of issue #386: higher-order derivatives of real numbers with respect to one variable
    const double a = point();
    real4th x = a;
    const auto d = derivatives(fsinx<real4th>, wrt(x), at(x));
    const double s = ::sin(a), c = ::cos(a);
    CHECK_APPROX(d[0], s * a);
    CHECK_APPROX(d[1], c * a + s);
    CHECK_APPROX(d[2], 2 * c - s * a);
    CHECK_APPROX(d[3], -3 * s - c * a);
    CHECK_APPROX(d[4], -4 * c + s * a);
}

AUTODIFF_NVRTC_TEST(issues_dual_along)
{
    const double a = point();
    dual4th x = a;
    real4th xr = a;
    const auto d = derivatives(fsinx<dual4th>, along(2.0), at(x));
    const auto dr = derivatives(fsinx<real4th>, along(2.0), at(xr));
    for(int i = 0; i < 5; ++i)
        CHECK_APPROX(d[i], dr[i]);
    auto g = taylorseries(fsinx<dual4th>, along(1.0), at(x));
    CHECK_APPROX_TOL(g(0.01), ::sin(a + 0.01) * (a + 0.01), 1e-11);
}

AUTODIFF_NVRTC_TEST(issues_inverse_hyperbolic)
{
    const double a = point();
    real x = a;
    CHECK_APPROX(derivative([](real x) -> real { return asinh(tan(x)); }, wrt(x), at(x)), 1.0 / ::cos(a));
    CHECK_APPROX(derivative([](real x) -> real { return atanh(2.0 * x); }, wrt(x), at(x)), 2.0 / (1.0 - 4 * a * a));
    CHECK_APPROX(derivative([](real x) -> real { return acosh(2.0 * x + 1.0); }, wrt(x), at(x)), 2.0 / ::sqrt((2 * a + 1) * (2 * a + 1) - 1));
}
