<p align="center">
  <a href="https://github.com/keckj/autodiff" target="_blank">
    <img src='art/cuda-autodiff-logo.svg' width='260'>
  </a>
</p>

# CUDA and NVRTC support

This fork of [autodiff](https://github.com/autodiff/autodiff) is based on
**autodiff v1.1.2** and makes the library usable in CUDA device code, both with
the offline compiler **nvcc** and with the runtime compiler **NVRTC**. The host
API is the one of upstream autodiff: code written for autodiff v1.1.2 compiles
unchanged with a regular C++17 compiler. The rest of this README, starting at
[Overview](#overview), is the upstream documentation.

The library stays header only. Functions are annotated with
`AUTODIFF_DEVICE_FUNC` (`__host__ __device__` with CUDA and HIP compilers,
empty otherwise) so that `dual`, `real` and `var` can be used in kernels, alone
or with Eigen vectors and matrices.

## What is supported on the device

- Forward mode with `dual` (all orders) and `real` (all orders), including
  `derivative`, `derivatives`, `gradient`, `jacobian`, `hessian`, directional
  derivatives (`along`) and Taylor series.
- Reverse mode with `var`: arithmetic, mathematical functions, conditions
  (`condition`, `min`, `max`, `!`, `&&`, `||`), `update`, `derivatives` and
  `derivativesx` (higher order).
- Eigen vectors and matrices of `dual`, `real` and `var`, with fixed and
  dynamic sizes (dynamic sizes allocate on the device heap).

`gradient`, `jacobian` and `hessian` return Eigen types with fixed sizes
(e.g. `Eigen::Vector3d` or `Eigen::Matrix<double, 2, 3>`) when the sizes of
all the variables given in `wrt(...)` are known at compile time, and dynamic
types (`Eigen::VectorXd`, `Eigen::MatrixXd`) otherwise. Code that assigns the
result to `Eigen::VectorXd` or `Eigen::MatrixXd` keeps working.

## Requirements

- A C++17 compiler for host code. Device code requires C++17 or C++20.
- CUDA 12.x. The library is developed and tested with CUDA 12.9 (nvcc and
  NVRTC, sm_120). Older toolkits have not been tested.
- Eigen 3.4 or Eigen 5 with nvcc (and on host). NVRTC requires Eigen headers
  patched for NVRTC (see below).

## Usage with nvcc

Include the autodiff headers in `.cu` files as usual, for instance:

```c++
#include <autodiff/forward/dual.hpp>
using namespace autodiff;

__global__ void kernel(const double* x, double* dfdx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    dual u = x[i];
    dfdx[i] = derivative([](dual u) -> dual { return u * sin(u); }, wrt(u), at(u));
}
```

The following nvcc options are required. They are added automatically when
linking against the CMake target `autodiff::autodiff`:

- `--expt-relaxed-constexpr`: autodiff calls `constexpr` functions of the
  standard library (`std::forward_as_tuple`, `std::get`, ...) from device
  code. Without this option nvcc only emits warnings (#20013, #20015) and the
  kernels silently compute wrong results;
- `--extended-lambda`, to pass `[] __host__ __device__ (...) { ... }` lambdas
  to `derivative`, `gradient`, ... (lambdas defined in device code do not need
  it).

The following options are recommended:

- `--diag-error=20011,20014` turns the calls of host functions from device
  functions into errors. Without it, nvcc only warns and the kernel aborts at
  runtime (possibly silently).
- `-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored` hides spurious
  warnings emitted when Eigen 3.4 is used in device code.

By default the library uses the `std` namespace for its utilities (`array`,
`tuple`, type traits, ...) with nvcc. Defining `AUTODIFF_USE_CUDA_STD=1` before
including any autodiff header makes it use `cuda::std` from libcu++ instead,
as with NVRTC.

## Usage with NVRTC

NVRTC has no C++ standard library: autodiff then uses libcu++ (`cuda::std`)
and the CUDA math API. The following options are required to compile a
program that includes autodiff:

- `-std=c++17` (or `-std=c++20`);
- `-default-device`: autodiff (and Eigen) rely on functions without execution
  space annotations, such as generic lambdas, that NVRTC otherwise considers
  as host functions and rejects. Without this option, the compilation fails
  in `autodiff/common/config.hpp` with an error naming the option;
- `-I<autodiff include directory>` and `-I<CUDA include directory>` (for the
  libcu++ headers);
- if Eigen is used: `-I<patched Eigen directory>` (see below). Defining
  `AUTODIFF_EIGEN_FOUND` is not needed: the Eigen overloads of `gradient`,
  `jacobian` and `hessian` are available whenever Eigen is included.

When autodiff is installed, its CMake package provides `autodiff_INCLUDE_DIR`
and `autodiff_NVRTC_OPTIONS` (`-std=c++17 -default-device
-I${autodiff_INCLUDE_DIR}`), to be completed with the CUDA include directory
(`CUDAToolkit_INCLUDE_DIRS` from `find_package(CUDAToolkit)`), and
`autodiff_NVRTC_EIGEN_PATCH_DIR`, the directory where the Eigen patch is
installed.

### Eigen with NVRTC

Upstream Eigen does not support NVRTC. The fork
[keckj/eigen-nvrtc](https://github.com/keckj/eigen-nvrtc) (release
[5.0.1-nvrtc](https://github.com/keckj/eigen-nvrtc/releases/tag/5.0.1-nvrtc))
is Eigen 5.0.1 with NVRTC support. The same changes are available as a patch
in [ext/eigen-nvrtc](ext/eigen-nvrtc), to be applied to an Eigen 5.0.1
checkout (see [ext/eigen-nvrtc/NVRTC.md](ext/eigen-nvrtc/NVRTC.md) for
details). All the changes are guarded by `__CUDACC_RTC__`: other compilers see
the same code as with Eigen 5.0.1. The CMake module
[cmake/AutodiffNvrtcEigen.cmake](cmake/AutodiffNvrtcEigen.cmake), used by the
NVRTC tests and examples, fetches the release at configure time; a directory
with already patched headers can be given with
`-DAUTODIFF_NVRTC_EIGEN_INCLUDE_DIR=...` instead.

With NVRTC, Eigen must be included first, before any autodiff header:

```c++
#include <Eigen/Core>
#include <autodiff/forward/dual/eigen.hpp>
```

## Reverse mode on the device

The expression trees of `var` are stored with reference counted pointers and
type erased functions. `std::shared_ptr` and `std::function` are host only,
so with CUDA compilers autodiff uses `autodiff::detail::shared_ptr` and
`autodiff::detail::function` from
[autodiff/common/memory.hpp](autodiff/common/memory.hpp) instead. This is
controlled by `AUTODIFF_VAR_USE_STD_MEMORY`: it defaults to 0 with nvcc, hipcc
and NVRTC, and to 1 otherwise; it can be overridden before including
autodiff, except with NVRTC where it is always 0.

This has the following consequences:

- The reference counts of `autodiff::detail::shared_ptr` are not atomic: an
  expression tree must not be shared between threads.
- Expression trees allocated on the device can only be used on the device, and
  conversely: `var` objects can not be passed as arguments of `__global__`
  functions nor copied between host and device. Pass `double` values to the
  kernels and create the variables in device code.
- The trees are allocated on the device heap and traversed recursively. Large
  expressions may require to increase the device heap size
  (`cudaDeviceSetLimit(cudaLimitMallocHeapSize, ...)`) or the stack size
  (`cudaDeviceSetLimit(cudaLimitStackSize, ...)`). A stack overflow is
  reported as `cudaErrorHardwareStackError` (714) at synchronization. ptxas
  warns that it can not determine the stack size statically because of the
  recursion: this warning is expected.

## Other limitations

- Stream output operators (`operator<<`) are host only with nvcc and are not
  available with NVRTC.
- Device code can not throw: errors that throw exceptions on host trigger an
  `assert` on the device.

## Changes with respect to autodiff v1.1.2

Besides the device annotations, this fork contains the following changes,
which also apply to host code:

- `copysign` is defined for `dual`, and `abs` of `real` no longer calls
  `std::copysign` (host only).
- `!condition` in reverse mode no longer refers to a destroyed temporary
  (dangling `this` capture in `BooleanExpr::operator!`).
- `derivatives(y, wrt(...))` in reverse mode no longer calls `std::array::fill`
  (host only).
- Compatibility with Eigen 5 (`Eigen::internal::SingleRange` became a
  template).
- `gradient`, `jacobian` and `hessian` return fixed size types when possible
  (see above).

It also fixes the following upstream bugs (regression tests in
[tests/issues.test.cpp](tests/issues.test.cpp) and
[tests/nvrtc/kernels/issues.test.cu](tests/nvrtc/kernels/issues.test.cu)):

- Forward and reverse modes can be included in the same compilation unit:
  they share the same `wrt` ([#179](https://github.com/autodiff/autodiff/issues/179)).
- `val(x)` of a non-const `real` no longer fails when the `dual` header is
  also included, and `eval(x)` is a no-op for `real` numbers
  ([#201](https://github.com/autodiff/autodiff/issues/201)).
- Higher-order derivatives of `real` numbers with respect to one variable,
  `derivatives(f, wrt(x), at(x))`, no longer fail to compile, and gradients
  can be computed with higher-order `real` numbers
  ([#201](https://github.com/autodiff/autodiff/issues/201),
  [#386](https://github.com/autodiff/autodiff/issues/386)).
- Directional derivatives of higher-order `dual` numbers,
  `derivatives(f, along(v), at(x))` and `taylorseries`, are no longer zero
  beyond the first order.
- `asinh`, `acosh` and `atanh` of `real` numbers apply the chain rule
  ([#395](https://github.com/autodiff/autodiff/issues/395),
  [#396](https://github.com/autodiff/autodiff/issues/396)). They are also
  available for `dual` and `var` numbers, which did not provide them.
- Matrices of `var` can be inverted and solved with Eigen 3.4
  ([#141](https://github.com/autodiff/autodiff/issues/141)): the conversion
  of `var` to `ExprPtr` is now explicit.
- The `gradient`, `jacobian` and `hessian` overloads returning Eigen types
  are available whenever Eigen is included, even without defining
  `AUTODIFF_EIGEN_FOUND` ([#372](https://github.com/autodiff/autodiff/issues/372)).

It finally adds math functions that were missing upstream, for the three
number types `dual`, `real` and `var` (on host, with nvcc and with NVRTC):

- `exp2`, `log2`, `expm1`, `log1p` and `erfc`;
- `fmin`, `fmax` and `fmod` (the value of `fmod` is exactly the one of
  `std::fmod`, its derivatives are 1 and minus the integral quotient);
- `floor`, `ceil`, `round` and `trunc`, with zero derivatives;
- the ones that only some types provided: `erf`, `hypot` (2 and 3 arguments),
  `abs2` and `conj` for `real`, `cbrt` for `dual` and `var`, `copysign` and
  `sign` for `var`.

## Building the tests and examples

```
git clone https://github.com/keckj/autodiff.git
cd autodiff
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build -j
./build/tests/autodiff-cpptests      # host tests
./build/tests/autodiff-cudatests     # nvcc tests, run on the device
./build/tests/nvrtc/autodiff-nvrtctests   # NVRTC tests, run on the device
```

The CUDA examples in [examples/cuda](examples/cuda) compute the same results
on the host, on the device with nvcc and on the device with NVRTC, and check
that they match. They are built with the other examples
(`-DAUTODIFF_BUILD_EXAMPLES=ON`) and can all be run with
`cmake --build build --target cuda-examples`.

The CUDA tests and examples are enabled when a CUDA compiler is found. The NVRTC test
runner can be disabled with `-DAUTODIFF_BUILD_NVRTC_TESTS=OFF`, and accepts
`--compile-only` to compile the kernels without a GPU.

<a href="https://autodiff.github.io" target="_blank">
    <img src='art/autodiff-header.svg' width='100%'>
</a>

![Linux build status](https://github.com/autodiff/autodiff/workflows/linux/badge.svg?branch=master)
![macOS build status](https://github.com/autodiff/autodiff/workflows/osx/badge.svg?branch=master)
![Windows build status](https://github.com/autodiff/autodiff/workflows/windows/badge.svg?branch=master)

# Overview

**autodiff** is a C++17 library that uses modern and advanced programming
techniques to enable automatic computation of derivatives in an efficient, easy,
and intuitive way.

## Demonstration

Consider the following function *f(x, y, z)*:

```c++
double f(double x, double y, double z)
{
    return (x + y + z) * exp(x * y * z);
}
```

which we use use to evaluate the variable *u = f(x, y, z)*:

```c++
double x = 1.0;
double y = 2.0;
double z = 3.0;
double u = f(x, y, z);
```

How can we minimally transform this code so that not only *u*, but also its
derivatives *∂u/∂x*, *∂u/∂y*, and *∂u/∂z*, can be computed?

The next two sections present how this can be achieved using two automatic
differentiation algorithms implemented in **autodiff**: **forward mode** and
**reverse mode**.

### Forward mode

In a *forward mode automatic differentiation* algorithm, both output variables
and one or more of their derivatives are computed together. For example, the
function evaluation *f(x, y, z)* can be transformed in a way that it will not
only produce the value of *u*, *the output variable*, but also one or more of
its derivatives *(∂u/∂x,  ∂u/∂y, ∂u/∂z)* with respect to the *input
variables* *(x, y, z)*.

Enabling forward automatic differentiation for the calculation of derivatives
using **autodiff** is relatively simple. For our previous function *f*, we only
need to replace the floating-point type `double` with `autodiff::dual` for both
input and output variables:

```c++
dual f(const dual& x, const dual& y, const dual& z)
{
    return (x + y + z) * exp(x * y * z);
}
```

We can now compute the derivatives *∂u/∂x*,  *∂u/∂y*, and *∂u/∂z* as follows:

```c++
dual x = 1.0;
dual y = 2.0;
dual z = 3.0;
dual u = f(x, y, z);

double dudx = derivative(f, wrt(x), at(x, y, z));
double dudy = derivative(f, wrt(y), at(x, y, z));
double dudz = derivative(f, wrt(z), at(x, y, z));
```

The auxiliary function `autodiff::wrt`, an acronym for **with respect to**,
is used to indicate which input variable *(x, y, z)* is the selected one to
compute the partial derivative of *f*. The auxiliary function `autodiff::at`
is used to indicate where (at which values of its parameters) the derivative
of *f* is evaluated.

### Reverse mode

In a *reverse mode automatic differentiation* algorithm, the output variable of
a function is evaluated first. During this function evaluation, all
mathematical operations between the input variables are *"recorded"* in an
*expression tree*. By traversing this tree from top-level (output variable as
the root node) to bottom-level (input variables as the leaf nodes), it is
possible to compute the contribution of each branch on the derivatives of the
output variable with respect to input variables.

<img
    src='art/expression-tree-diagram.svg'
    style='max-width:100%;'
    title='Expression tree diagram.'>

Thus, a single pass in a reverse mode calculation **computes all derivatives**,
in contrast with forward mode, which requires one pass for each input variable.
Note, however, that it is possible to change the behavior of a forward pass so
that many (perhaps even all) derivatives of an output variable are computed
simultaneously (e.g., in a single forward pass, *∂u/∂x*,  *∂u/∂y*, and *∂u/∂z*
are evaluated together with *u*, in contrast with three forward passes, each
one computing the individual derivatives).

Similar as before, we can use **autodiff** to enable reverse automatic
differentiation for our function *f* by simply replacing type `double` with
`autodiff::var` as follows:

```c++
var f(var x, var y, var z)
{
    return (x + y + z) * exp(x * y * z);
}
```

The code below demonstrates how the derivatives *∂u/∂x*,  *∂u/∂y*, and *∂u/∂z*
can be calculated:

```c++
var x = 1.0;
var y = 2.0;
var z = 3.0;
var u = f(x, y, z);

Derivatives dud = derivatives(u);

double dudx = dud(x);
double dudy = dud(y);
double dudz = dud(z);
```

The function `autodiff::derivatives` will traverse the expression tree stored
in variable `u` and compute all its derivatives with respect to the input
variables *(x, y, z)*, which are then stored in the object `dud`. The
derivative of `u` with respect to input variable `x` (i.e., *∂u/∂x*) can then
be extracted from `dud` using `dud(x)`. The operations `dud(x)`, `dud(y)`,
`dud(z)` involve no computations! Just extraction of derivatives previously
computed with a call to function `autodiff::derivatives`.

## Documentation

Check the documentation website for more details:

<a href="https://autodiff.github.io" target="_blank">
    <img src='art/autodiff.github.io.svg' width='100%'>
</a>

# License

MIT License

Copyright © 2018–2024 Allan Leal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

[issues]: https://github.com/autodiff/autodiff/issues/new
