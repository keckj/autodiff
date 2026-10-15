// Minimal device-side test framework for kernels compiled with NVRTC by the autodiff-nvrtctests runner.
//
// Each test is declared with AUTODIFF_NVRTC_TEST(name) { ... } and becomes an extern "C" kernel. The runner finds the
// tests by scanning the source for AUTODIFF_NVRTC_TEST(...) and launches each one with a counter of failed checks.

#pragma once

#define AUTODIFF_NVRTC_TEST(name) extern "C" __global__ void name(unsigned int* __autodiff_nfailures)

#define AUTODIFF_NVRTC_FAIL(fmt, ...)                                                                  \
    do {                                                                                               \
        printf("%s:%d: thread %d: " fmt "\n", __FILE__, __LINE__, (int)threadIdx.x, __VA_ARGS__);       \
        atomicAdd(__autodiff_nfailures, 1u);                                                           \
    } while(0)

#define CHECK(cond)                                                                                    \
    do {                                                                                               \
        if(!(cond))                                                                                    \
            AUTODIFF_NVRTC_FAIL("CHECK(%s) failed", #cond);                                            \
    } while(0)

#define CHECK_APPROX_TOL(a, b, tol)                                                                    \
    do {                                                                                               \
        const double __a = static_cast<double>(a);                                                     \
        const double __b = static_cast<double>(b);                                                     \
        const double __d = __a > __b ? __a - __b : __b - __a;                                          \
        const double __m = (__b > 0 ? __b : -__b) > 1.0 ? (__b > 0 ? __b : -__b) : 1.0;                 \
        if(!(__d <= (tol) * __m))                                                                      \
            AUTODIFF_NVRTC_FAIL("CHECK_APPROX(%s, %s) failed: %.17g != %.17g", #a, #b, __a, __b);      \
    } while(0)

#define CHECK_APPROX(a, b) CHECK_APPROX_TOL(a, b, 1e-12)
