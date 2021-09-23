<a href="https://autodiff.github.io" target="_blank">
    <img src='art/cuda-autodiff-logo.svg' width='100%'>
</a>

# Overview of cuda autodiff
Cuda-autodiff is a fork of [autodiff](https://github.com/autodiff/autodiff) thats has been ported to CUDA 11. It is currently based against **autodiff version 0.6.4+** (commit 9cd6af0) and supports a subset of the autodiff library with some limitations due to the fact that it has to run in device code. The library and its dependencies are header only and can be included directly in kernel code. Most of the routines are implemented as \_\_host\_\_ \_\_device\_\_ when possible, see the current limitations subsection. All tests and examples adapted from the original library are running on both host and device.

Even though cuda now support heap allocation in \_\_device_\_ code, the autodiff library internals have been completely reworked to handle only statically sized vector and matrix types. This choice has been made for performance but also because of the lack of support of dynamic allocation in cuda dependencies. Eigen3 in cuda kernels does not support dynamically sized types and the [cuda C++ standard library](https://nvidia.github.io/libcudacxx/) does not provide `std::vector`.

## How to get started
Look at official [autodiff documentation](https://autodiff.github.io/) and compare autodiff examples to cuda-autodiff examples in the examples directory. It is easier to get started with forward mode autodiff because there are less device-side limitations to consider.

```
git clone --recursive https://github.com/keckj/autodiff.git
cd autodiff
mkdir build
cd build
cmake ..
make
./tests/autodiff-cpptests
./examples/forward/example-forward-single-variable-function
```

## Dependencies
Bellow are the minimul required version supported by cuda-autodiff:
 - CUDA 11, both nvcc and nvrtc are supported
 - libcu++ 1.3.0 (shipped with cuda 11.2, header only)
 - Eigen 3.4 (header only, available in the [ext/eigen](ext/eigen) directory)

Note that `std::complex` support can be added by using libcu++ 1.4.1 or later (shipped with cuda 11.3+).
The library is currently developped and tested under CUDA 11.4

## How to use the library
The library and all of its dependencies are header only. The following build options may be required:
1. `-I/path/to/autodiff` path to cuda-autodiff headers.
3. `-I/path/to/eigen` path to eigen3 headers.
2. `-I/usr/local/cuda/include` or equivalent path to your libcu++ headers.
3. `--diag_suppress=esa_on_defaulted_function_ignored` if you want to hide some spurious eigen warnings.

Cuda-autodiff is a fork of autodiff. It currently use the same namespace and may thus clash
with existing autodiff installations on your system. See `CMakelists.txt` for an example of cmake integration.

## What is currently supported
 - Forward mode autodiff using  `real` and `dual` (see [examples/forward](examples/forward)).
 - Reverse mode autodiff using `var` as long as variables are not passed as _\_global\_\_ function arguments (see [examples/backward](examples/backward)).
 - All tests and examples and tests from the original autodiff library have been adapted to work with cuda.
 - Support for `std::array` like types based on a distinct `cuda::std::array` type.
 - Eigen3 statically sized types (`Array` and `Matrix`).
 - Custom user-provided scalar and vector types (statically allocated with compile-time known size).

## What will never be included
 - `std::vector` based types (dynamic allocation).
 - Eigen3 dynamically resizable types (Array and Matrix with dynamic sizes).
 - Custom user-provided scalar and vector types that perform heap allocations.


## Current limitations
- Forward mode autodiff will work as expected.
- Reverse mode autodiff exhibit some limitations due to the use of virtual inheritance and shared pointer to build the expression tree. This leads to the following device-side restrictions:
    1. Types used in reversed mode autodiff such as `var` cannot be used as \_\_global\_\_ function arguments (it is not allowed to pass as an argument to a global function an object of a class with virtual functions). See [examples/reverse/example-reverse-multi-variable-function.cu](examples/reverse/example-reverse-multi-variable-function.cu).
    2. Ptxas is not able to statically determine maximum stack size when using reverse mode autodiff on the device and will emit a warning about it. A stack overflow may occur with default device stack limit when using large expressions. In this case, you will get  `cudaErrorHardwareStackError (714)` on stream synchronization. Maximum device supported stack size can be retrieved with `cudaDeviceGetLimit(size_t *pValue, cudaLimitStackSize)` and effective device limit can be set with `cudaDeviceSetLimit(new_limit, cudaLimitStackSize)`. See [examples/reverse/example-reverse-hessian-derivatives-using-eigen.cu](examples/reverse/example-reverse-hessian-derivatives-using-eigen.cu).
    3. The expression tree is built dynamically by using heap allocations. If required, maximum device supported heap size can be retrieved with `cudaDeviceGetLimit(size_t *pValue, cudaLimitMallocHeapSize)` and effective device limit can be set with `cudaDeviceSetLimit(new_limit, cudaLimitMallocHeapSize)`.

- `std::stream` (ostream, stringstream) based functions are only defined as \_\_host\_\_ functions when using nvcc and are disabled when using nvrtc.
- Some spurious warnings may appear when using eigen. They can be removed by providing the following compiler option: `--diag_suppress=esa_on_defaulted_function_ignored`.

## Differences with original autodiff library
- Occurences of `std::vector<T>` have been replaced with `cuda::std::array<T,N>`.
- Dynamically sized types such as `autodiff::VectorX`, `autodiff::MatrixX` and `autodiff::ArrayX` have been replaced by statically sized types `autodiff::Vector<T,N>`, `autodiff::Matrix<T,N,M>` and `autodiff::Array<T,N,M=1>`. Those types are aliases of static eigen3 types Matrix and Array and work in device code.
- As `cuda::std::shared_ptr` is not provided by the standard c++ cuda library, `autodiff::detail::shared_ptr` has been implemented in `autodiff/common/memory.hpp` to cover basic shared pointer usage in cuda-autodiff. All dynamic casts have been removed from the library because \_\_device\_\_ functions cannot call `dynamic_cast` or `dynamic_pointer_cast` (RTTI-related features are supported in host code, but not in device code).
- As `cuda::std::array` has not yet been implemented in the standard c++ cuda library (libcu++, see this [issue](https://github.com/NVIDIA/libcudacxx/issues/51)), a basic implementation is provided by the cuda-autodiff library. Everything is supported as long as array iterators are not involved. Please note that `std::array` and `cuda::std::array` represent distinct types, cuda-autodiff internally works with `cuda::std::array` which is available in the `<autodiff/std/array>` header but lives in the `cuda::std` namespace.



# General information about autodiff v0.6.4
<a href="https://autodiff.github.io" target="_blank">
    <img src='art/autodiff-logo.svg' width='100%'>
</a>

[Autodiff](https://github.com/autodiff/autodiff) is a C++17 library that uses modern and advanced programming
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

## Development status

**autodiff** is planned to be a long-term maintained automatic differentiation
C++ project. This means that more algorithms and number types will still be
implemented and that the currently existing ones will be further enhanced.
Please have in mind, however, that **autodiff** is still in a relatively early
stage of development, which implies that minor breaking changes in API may be
introduced to simplify its use and make it more intuitive and consistent with
new library additions.

## Documentation

Check the documentation website for more details:

<a href="https://autodiff.github.io" target="_blank">
    <img src='art/autodiff.github.io.svg' width='100%'>
</a>

# Licenses

This repository is licensed as the following:
1. This repository contains a substancial portion of the original autodiff library that is licensed under the MIT license.

2. The header that provides array implementation (autodiff/std/array) is part of the LLVM Project, and is licensed under the Apache License v2.0 with LLVM Exceptions (https://llvm.org/LICENSE.txt).

3. Apart from this header, all other modifications and additions to the original autodiff codebase constitute the cuda-autodiff library and is licensed under the MIT license.

## Cuda-autodiff license
MIT Licence

Copyright (c) 2021 Jean-Baptiste Keck

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

## Autodiff license

MIT Licence

Copyright (c) 2018–2021 Allan Leal

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
