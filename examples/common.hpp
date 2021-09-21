#include <limits>
#include <autodiff/common/eigen.hpp>
#include <autodiff/forward/real.hpp>

namespace autodiff {

const double eps = std::numeric_limits<double>::epsilon();

__host__ bool allclose(double x, double y, size_t n=1)
{
    const double diff = std::fabs(x - y);
    if (diff <= n*eps) {
        return true;
    }
    if (diff <= n*eps*std::fmax(std::fabs(x), std::fabs(y))) {
        return true;
    }
    return false;
}

template<int N, typename Scalar>
__host__ bool allclose(const detail::Real<N,Scalar>& x, const detail::Real<N,Scalar>& y, size_t n=1) {
    for (size_t j = 0; j < N; j++) {
        if (!allclose(x[j], y[j], n)) {
            return false;
        }
    }
    return true;
}

template<typename Scalar, int M, int N>
__host__ bool allclose(const Array<Scalar,M,N>& x, const Array<Scalar,M,N>& y, size_t n=1) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            if (!allclose(x(i,j), y(i,j), n)) {
                return false;
            }
        }
    }
    return true;
}

template<typename Scalar, int M, int N>
__host__ bool allclose(const Matrix<Scalar,M,N>& x, const Matrix<Scalar,M,N>& y, size_t n=1) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            if (!allclose(x(i,j), y(i,j), n)) {
                return false;
            }
        }
    }
    return true;
}

} /* end of namespace autodiff */
