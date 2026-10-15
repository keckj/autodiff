// Device tests (NVRTC) of forward mode automatic differentiation with Eigen vectors and matrices of dual and real
// numbers. Requires Eigen headers patched for NVRTC (see ext/eigen-nvrtc/NVRTC.md).

// Eigen must be included first, before any header using std:: (see ext/eigen-nvrtc/NVRTC.md)
#include <Eigen/Core>

#include <nvrtc_test.hpp>

#include <autodiff/forward/dual/eigen.hpp>
#include <autodiff/forward/real/eigen.hpp>

using namespace autodiff;

namespace {

// Each thread evaluates the functions at a different point
__device__ double point() { return 0.1 + 0.01 * threadIdx.x; }

template<typename V>
__device__ auto f3(const V& x) -> typename V::Scalar
{
    // f(x) = x0 x1 x2 + |x|^2 + exp(x0)
    return x[0] * x[1] * x[2] + x.squaredNorm() + exp(x[0]);
}

} // namespace

AUTODIFF_NVRTC_TEST(eigen_dual_linear_algebra)
{
    const double a = point();
    Matrix3dual A;
    A << 1, 2, 3, 4, 5, 6, 7, 8, a;
    Vector3dual x(a, 2 * a, 3 * a);

    const Vector3dual y = A * x;
    CHECK_APPROX(val(y[0]), 14 * a);
    CHECK_APPROX(val(y[2]), 7 * a + 16 * a + 3 * a * a);
    CHECK_APPROX(val(x.dot(x)), 14 * a * a);
    CHECK_APPROX(val(A.transpose()(2, 0)), 3.0);
    CHECK_APPROX(val(A.trace()), 6 + a);
    CHECK_APPROX(val(x.array().exp()[1]), ::exp(2 * a));
    CHECK_APPROX(val(x.cwiseProduct(x).sum()), 14 * a * a);

    const Eigen::Vector3d xd = x.cast<double>();
    CHECK_APPROX(xd[2], 3 * a);
}

AUTODIFF_NVRTC_TEST(eigen_dual_gradient)
{
    const double a = point();
    Vector3dual x(a, 2.0, 3.0);
    dual u;
    const auto g = gradient(f3<Vector3dual>, wrt(x), at(x), u);
    static_assert(stdx::is_same_v<stdx::remove_cv_t<decltype(g)>, Eigen::Vector3d>);
    CHECK_APPROX(val(u), a * 6.0 + a * a + 13.0 + ::exp(a));
    CHECK_APPROX(g[0], 6.0 + 2 * a + ::exp(a));
    CHECK_APPROX(g[1], 3 * a + 4.0);
    CHECK_APPROX(g[2], 2 * a + 6.0);

    // Gradient with respect to a single entry and a mixture of scalars and vectors
    dual s = a;
    auto h = [](const Vector3dual& x, const dual& s) -> dual { return s * x.sum() + s * s; };
    const auto gm = gradient(h, wrt(s, x), at(x, s));
    static_assert(stdx::is_same_v<stdx::remove_cv_t<decltype(gm)>, Eigen::Vector4d>);
    CHECK_APPROX(gm[0], a + 5.0 + 2 * a);
    CHECK_APPROX(gm[1], a);
    CHECK_APPROX(gm[3], a);
}

AUTODIFF_NVRTC_TEST(eigen_dual_jacobian)
{
    const double a = point();
    Vector3dual x(a, 2.0, 3.0);
    auto F = [](const Vector3dual& x) -> Vector2dual {
        Vector2dual r;
        r << x[0] * x[1], sin(x[2]) + x[0];
        return r;
    };
    Vector2dual Fx;
    const auto J = jacobian(F, wrt(x), at(x), Fx);
    static_assert(stdx::is_same_v<stdx::remove_cv_t<decltype(J)>, Eigen::Matrix<double, 2, 3>>);
    CHECK_APPROX(val(Fx[0]), 2 * a);
    CHECK_APPROX(val(Fx[1]), ::sin(3.0) + a);
    CHECK_APPROX(J(0, 0), 2.0);
    CHECK_APPROX(J(0, 1), a);
    CHECK_APPROX(J(0, 2), 0.0);
    CHECK_APPROX(J(1, 0), 1.0);
    CHECK_APPROX(J(1, 1), 0.0);
    CHECK_APPROX(J(1, 2), ::cos(3.0));
}

AUTODIFF_NVRTC_TEST(eigen_dual_hessian)
{
    const double a = point();
    Vector3dual2nd x(a, 2.0, 3.0);
    dual2nd u;
    Eigen::Vector3d g;
    const auto H = hessian(f3<Vector3dual2nd>, wrt(x), at(x), u, g);
    static_assert(stdx::is_same_v<stdx::remove_cv_t<decltype(H)>, Eigen::Matrix3d>);
    CHECK_APPROX(g[0], 6.0 + 2 * a + ::exp(a));
    CHECK_APPROX(H(0, 0), 2.0 + ::exp(a));
    CHECK_APPROX(H(0, 1), 3.0);
    CHECK_APPROX(H(1, 0), 3.0);
    CHECK_APPROX(H(0, 2), 2.0);
    CHECK_APPROX(H(1, 2), a);
    CHECK_APPROX(H(2, 1), a);
    CHECK_APPROX(H(1, 1), 2.0);
    CHECK_APPROX(H(2, 2), 2.0);
}

AUTODIFF_NVRTC_TEST(eigen_real_gradient_jacobian)
{
    const double a = point();
    Vector3real x(a, 2.0, 3.0);
    real u;
    const auto g = gradient(f3<Vector3real>, wrt(x), at(x), u);
    static_assert(stdx::is_same_v<stdx::remove_cv_t<decltype(g)>, Eigen::Vector3d>);
    CHECK_APPROX(val(u), a * 6.0 + a * a + 13.0 + ::exp(a));
    CHECK_APPROX(g[0], 6.0 + 2 * a + ::exp(a));
    CHECK_APPROX(g[1], 3 * a + 4.0);
    CHECK_APPROX(g[2], 2 * a + 6.0);

    auto F = [](const Vector3real& x) -> Vector2real {
        Vector2real r;
        r << x[0] * x[1], x[2] * x[2];
        return r;
    };
    const auto J = jacobian(F, wrt(x), at(x));
    static_assert(stdx::is_same_v<stdx::remove_cv_t<decltype(J)>, Eigen::Matrix<double, 2, 3>>);
    CHECK_APPROX(J(0, 0), 2.0);
    CHECK_APPROX(J(0, 1), a);
    CHECK_APPROX(J(1, 2), 6.0);
    CHECK_APPROX(J(1, 0), 0.0);

    // Directional derivatives of higher order along an Eigen vector
    Vector2real4th v(a, 2 * a);
    const Eigen::Vector2d dir(1.0, -1.0);
    auto p = [](const Vector2real4th& v) -> real4th { return v[0] * v[1]; };
    const auto d = derivatives(p, along(dir), at(v));
    CHECK_APPROX(d[0], 2 * a * a);
    CHECK_APPROX(d[1], a);
    CHECK_APPROX(d[2], -2.0);
}

AUTODIFF_NVRTC_TEST(eigen_dual_dynamic)
{
    // Dynamic sizes allocate on the device heap
    const double a = point();
    VectorXdual x(4);
    x << a, 1.0, 2.0, 3.0;
    auto f = [](const VectorXdual& x) -> dual { return x.squaredNorm(); };
    const Eigen::VectorXd g = gradient(f, wrt(x), at(x));
    CHECK(g.size() == 4);
    CHECK_APPROX(g[0], 2 * a);
    CHECK_APPROX(g[3], 6.0);
}
