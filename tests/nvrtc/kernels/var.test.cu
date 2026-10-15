// Device tests (NVRTC) of reverse mode automatic differentiation with autodiff::var. The expression trees are allocated
// on the device heap and evaluated with virtual functions (see AUTODIFF_VAR_USE_STD_MEMORY).

// Eigen must be included first, before any header using std:: (see ext/eigen-nvrtc/NVRTC.md)
#include <Eigen/Core>

#include <nvrtc_test.hpp>

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

using namespace autodiff;

namespace {

// Each thread evaluates the functions at a different point in (0.1, 0.42)
__device__ double point() { return 0.1 + 0.01 * threadIdx.x; }

__device__ double grad(const var& y, var& x) { return derivatives(y, wrt(x))[0]; }

// Count the destructions of the objects owned by detail::shared_ptr
struct Counted
{
    int* ndestroyed;
    int value;
    __device__ Counted(int* n, int v) : ndestroyed(n), value(v) {}
    __device__ virtual ~Counted() { ++*ndestroyed; }
};

struct CountedDerived : Counted
{
    __device__ CountedDerived(int* n, int v) : Counted(n, v) {}
};

} // namespace

AUTODIFF_NVRTC_TEST(var_arithmetic)
{
    const double a = point();
    var x = a;
    var y = 2.0;

    var f = (x + 1) * (x - y) / (2 * y) + x * x * x - 3 / x;
    const double fx = (2 * a + 1 - 2.0) / (2 * 2.0) + 3 * a * a + 3 / (a * a);
    const double fy = (a + 1) * (-2.0 * 2.0 - 2 * (a - 2.0)) / (4 * 2.0 * 2.0);
    const auto [dfx, dfy] = derivatives(f, wrt(x, y));
    CHECK_APPROX(dfx, fx);
    CHECK_APPROX(dfy, fy);

    var z = x;
    z += y; z -= 1.0; z *= x; z /= y;
    CHECK_APPROX(val(z), (a + 2.0 - 1.0) * a / 2.0);
    CHECK(x < y && y > x && x <= x && x != y && x == a);
    CHECK_APPROX(val(-x), -a);
    CHECK_APPROX(grad(-x, x), -1.0);
}

AUTODIFF_NVRTC_TEST(var_functions)
{
    const double a = point();
    var x = a;
    var y = 2 * a + 0.5;

    CHECK_APPROX(grad(sin(x), x), ::cos(a));
    CHECK_APPROX(grad(cos(x), x), -::sin(a));
    CHECK_APPROX(grad(tan(x), x), 1.0 / (::cos(a) * ::cos(a)));
    CHECK_APPROX(grad(asin(x), x), 1.0 / ::sqrt(1 - a * a));
    CHECK_APPROX(grad(acos(x), x), -1.0 / ::sqrt(1 - a * a));
    CHECK_APPROX(grad(atan(x), x), 1.0 / (1 + a * a));
    CHECK_APPROX(grad(sinh(x), x), ::cosh(a));
    CHECK_APPROX(grad(cosh(x), x), ::sinh(a));
    CHECK_APPROX(grad(tanh(x), x), 1.0 / (::cosh(a) * ::cosh(a)));
    CHECK_APPROX(grad(asinh(x), x), 1.0 / ::sqrt(a * a + 1));
    CHECK_APPROX(grad(acosh(x + 1), x), 1.0 / ::sqrt((a + 1) * (a + 1) - 1));
    CHECK_APPROX(grad(atanh(x), x), 1.0 / (1 - a * a));
    CHECK_APPROX(grad(exp(x), x), ::exp(a));
    CHECK_APPROX(grad(log(x), x), 1.0 / a);
    CHECK_APPROX(grad(log10(x), x), 1.0 / (a * ::log(10.0)));
    CHECK_APPROX(grad(sqrt(x), x), 0.5 / ::sqrt(a));
    CHECK_APPROX(grad(pow(x, 3.0), x), 3 * a * a);
    CHECK_APPROX(grad(pow(x, y), y), ::pow(a, 2 * a + 0.5) * ::log(a));
    CHECK_APPROX(grad(abs(-x), x), 1.0);
    CHECK_APPROX(grad(abs2(x), x), 2 * a);
    CHECK_APPROX(grad(erf(x), x), 2.0 / ::sqrt(3.14159265358979323846) * ::exp(-a * a));
    CHECK_APPROX(grad(log2(x), x), 1.0 / (a * ::log(2.0)));
    CHECK_APPROX(grad(log1p(x), x), 1.0 / (1 + a));
    CHECK_APPROX(grad(exp2(x), x), ::log(2.0) * ::exp2(a));
    CHECK_APPROX(grad(expm1(x), x), ::exp(a));
    CHECK_APPROX(grad(cbrt(x), x), 1.0 / (3 * ::cbrt(a * a)));
    CHECK_APPROX(grad(erfc(x), x), -2.0 / ::sqrt(3.14159265358979323846) * ::exp(-a * a));
    CHECK_APPROX(grad(floor(x) + ceil(x) + round(x) + trunc(x), x), 0.0);
    CHECK_APPROX(val(floor(x + 1)), 1.0);
    CHECK(val(fmod(y, x)) == ::fmod(val(y), a));
    CHECK_APPROX(grad(fmod(y, x), y), 1.0);
    CHECK_APPROX(grad(fmod(y, x), x), -::round((val(y) - ::fmod(val(y), a)) / a));
    CHECK_APPROX(grad(fmin(x, y), x), 1.0);
    CHECK_APPROX(grad(fmax(x, y), y), 1.0);
    CHECK_APPROX(grad(copysign(x, -y), x), -1.0);
    CHECK_APPROX(val(sign(-x)), -1.0);
    CHECK_APPROX(grad(atan2(x, y), x), val(y) / (a * a + val(y) * val(y)));
    CHECK_APPROX(grad(hypot(x, y), x), a / ::hypot(a, val(y)));
    CHECK_APPROX(val(hypot(x, y, 3.0)), ::sqrt(a * a + val(y) * val(y) + 9.0));
    CHECK_APPROX(grad(hypot(x, y, 3.0), x), a / ::sqrt(a * a + val(y) * val(y) + 9.0));
}

AUTODIFF_NVRTC_TEST(var_conditions)
{
    const double a = point();
    var x = a;
    var y = 0.25;

    var m = min(x, y);
    var M = max(x, y);
    CHECK_APPROX(val(m), a < 0.25 ? a : 0.25);
    CHECK_APPROX(grad(m, x), a < 0.25 ? 1.0 : 0.0);
    CHECK_APPROX(grad(M, x), a > 0.25 ? 1.0 : 0.0);
    CHECK_APPROX(val(sgn(x - y)), a > 0.25 ? 1.0 : (a < 0.25 ? -1.0 : 0.0));

    var c = condition(x < y, x * x, 3 * x);
    CHECK_APPROX(grad(c, x), a < 0.25 ? 2 * a : 3.0);

    // Updating an independent variable reevaluates the expressions depending on it
    x.update(1.0);
    c.update();
    CHECK_APPROX(val(c), 3.0);
    CHECK_APPROX(grad(c, x), 3.0);

    var n = condition(!(x < y), x * x, 3 * x);
    CHECK_APPROX(val(n), 1.0);
    x.update(0.1);
    n.update();
    CHECK_APPROX(val(n), 0.3);
    CHECK_APPROX(grad(n, x), 3.0);
}

AUTODIFF_NVRTC_TEST(var_higher_order)
{
    const double a = point();
    var x = a;
    var y = 2.0;
    var f = x * x * y + sin(x);

    const auto [fx, fy] = derivativesx(f, wrt(x, y));
    CHECK_APPROX(val(fx), 2 * a * 2.0 + ::cos(a));
    CHECK_APPROX(val(fy), a * a);
    const auto [fxx, fxy] = derivatives(fx, wrt(x, y));
    CHECK_APPROX(fxx, 2 * 2.0 - ::sin(a));
    CHECK_APPROX(fxy, 2 * a);
}

AUTODIFF_NVRTC_TEST(var_eigen)
{
    const double a = point();
    Vector3var x(a, 2.0, 3.0);
    var y = x[0] * x[1] * x[2] + x.squaredNorm();

    Eigen::Vector3d g = gradient(y, x);
    CHECK_APPROX(g[0], 6.0 + 2 * a);
    CHECK_APPROX(g[1], 3 * a + 4.0);
    CHECK_APPROX(g[2], 2 * a + 6.0);

    Eigen::VectorXd gx;
    const auto H = hessian(y, x, gx);
    CHECK_APPROX(gx[0], 6.0 + 2 * a);
    CHECK_APPROX(H(0, 0), 2.0);
    CHECK_APPROX(H(0, 1), 3.0);
    CHECK_APPROX(H(1, 2), a);
    CHECK_APPROX(H(2, 1), a);

    VectorXvar z(4);
    z << a, 1.0, 2.0, 3.0;
    var s = z.sum() * z[0];
    const Eigen::VectorXd gz = gradient(s, z);
    CHECK(gz.size() == 4);
    CHECK_APPROX(gz[0], (a + 6.0) + a);
    CHECK_APPROX(gz[3], a);
}

AUTODIFF_NVRTC_TEST(var_memory)
{
    using detail::shared_ptr;
    using detail::make_shared;

    int ndestroyed = 0;
    {
        shared_ptr<Counted> p = make_shared<CountedDerived>(&ndestroyed, 7);
        CHECK(p.use_count() == 1);
        {
            shared_ptr<Counted> q = p;
            shared_ptr<Counted> r;
            r = q;
            CHECK(p.use_count() == 3);
            auto d = detail::static_pointer_cast<CountedDerived>(r);
            CHECK(p.use_count() == 4 && d->value == 7);
            shared_ptr<Counted> m = stdx::move(q);
            CHECK(!q && p.use_count() == 4);
        }
        CHECK(p.use_count() == 1 && ndestroyed == 0);
        p.reset();
        CHECK(!p && ndestroyed == 1);
    }
    CHECK(ndestroyed == 1);

    detail::function<int(int)> f = [k = 3](int i) { return k * i; };
    detail::function<int(int)> g = f;
    CHECK(f(2) == 6 && g(5) == 15);

    // Expression trees are released once no variable refers to them
    const double a = point();
    var x = a;
    CHECK(x.expr.use_count() == 1);
    {
        var y = sin(x) * x + exp(x / 1000.0) - x * x;
        CHECK(x.expr.use_count() > 1);
        CHECK_APPROX(derivatives(y, wrt(x))[0], ::cos(a) * a + ::sin(a) + ::exp(a / 1000.0) / 1000.0 - 2 * a);
    }
    CHECK(x.expr.use_count() == 1);
    double acc = 0.0;
    for(int i = 0; i < 200; ++i) {
        var z = x + i;
        var w = sin(z) * z;
        acc += derivatives(w, wrt(x))[0];
    }
    CHECK(acc == acc && x.expr.use_count() == 1);
}
