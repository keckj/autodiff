//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright © 2018–2024 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

//======================================================================================================================
// Harness of the CUDA examples of autodiff.
//
// Each example computes the same quantities three times:
//   - on the host, with the code compiled by the host compiler (through nvcc);
//   - on the device, with a kernel compiled by nvcc (see runOnDevice);
//   - on the device, with the same kernel compiled at runtime by NVRTC (see runWithNvrtc).
// The three results are then compared with allclose (see exitExample).
//
// The code compiled by NVRTC is the part of the example file located between the lines AUTODIFF_NVRTC_BEGIN_SOURCE
// and AUTODIFF_NVRTC_END_SOURCE. These macros expand to nothing, so that this code is also compiled by nvcc. It must
// only use what is available with NVRTC: no C++ standard library (autodiff then uses libcu++, see stdx in
// autodiff/common/stdlib.hpp), no I/O of Eigen types, and when Eigen is used, <Eigen/Core> must be included first,
// before any header using std:: (see ext/eigen-nvrtc/NVRTC.md).
//
// Kernels take their inputs by value and write their (trivially copyable) results to their last argument:
//
//     __global__ void kernel(Input1 a, Input2 b, Results* output);
//
// Expression trees of autodiff::var created on the device live on the device heap: they can not be passed between
// the host and the device, so kernels using var take plain numbers and return plain numbers.
//
// The include directories given to NVRTC are passed as compile definitions by CMake (see examples/cuda/CMakeLists.txt):
// AUTODIFF_EXAMPLES_AUTODIFF_INCLUDE_DIR, AUTODIFF_EXAMPLES_EIGEN_INCLUDE_DIR and AUTODIFF_EXAMPLES_CUDA_INCLUDE_DIR.
//======================================================================================================================

// C++ includes
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

// Eigen includes
#include <Eigen/Core>

#ifndef AUTODIFF_EXAMPLES_AUTODIFF_INCLUDE_DIR
#error "AUTODIFF_EXAMPLES_AUTODIFF_INCLUDE_DIR must be defined (directory containing autodiff/)"
#endif
#ifndef AUTODIFF_EXAMPLES_EIGEN_INCLUDE_DIR
#error "AUTODIFF_EXAMPLES_EIGEN_INCLUDE_DIR must be defined (directory containing the NVRTC compatible Eigen/Core)"
#endif
#ifndef AUTODIFF_EXAMPLES_CUDA_INCLUDE_DIR
#error "AUTODIFF_EXAMPLES_CUDA_INCLUDE_DIR must be defined (CUDA include directory)"
#endif

/// Marks the beginning of the code of an example file that is also compiled at runtime with NVRTC.
#define AUTODIFF_NVRTC_BEGIN_SOURCE
/// Marks the end of the code of an example file that is also compiled at runtime with NVRTC.
#define AUTODIFF_NVRTC_END_SOURCE

namespace autodiff {
namespace examples {

//======================================================================================================================
//
// ERROR HANDLING
//
//======================================================================================================================

[[noreturn]] inline void fail(const std::string& what, const std::string& error)
{
    std::cerr << "error: " << what << " failed: " << error << std::endl;
    std::exit(EXIT_FAILURE);
}

inline void check(cudaError_t res, const std::string& what)
{
    if(res != cudaSuccess)
        fail(what, std::string(cudaGetErrorName(res)) + ": " + cudaGetErrorString(res));
}

inline void check(CUresult res, const std::string& what)
{
    if(res != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* msg = nullptr;
        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);
        fail(what, std::string(name ? name : "unknown error") + ": " + (msg ? msg : ""));
    }
}

inline void check(nvrtcResult res, const std::string& what)
{
    if(res != NVRTC_SUCCESS)
        fail(what, nvrtcGetErrorString(res));
}

//======================================================================================================================
//
// DEVICE SETUP
//
//======================================================================================================================

/// Initialize the CUDA runtime and driver APIs on device 0 (once). Both APIs share the primary context of the device.
inline void initDevice()
{
    static bool initialized = false;
    if(initialized)
        return;
    check(cudaSetDevice(0), "cudaSetDevice");
    check(cudaFree(nullptr), "cudaFree"); // creates the primary context and makes it current
    // Reverse mode (autodiff::var) allocates its expression trees on the device heap and evaluates them recursively
    check(cudaDeviceSetLimit(cudaLimitMallocHeapSize, std::size_t(64) << 20), "cudaDeviceSetLimit(heap size)");
    check(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024), "cudaDeviceSetLimit(stack size)");
    check(cuInit(0), "cuInit");
    initialized = true;
}

/// Copy the results of a kernel from the device, then free the device memory.
template<typename Result>
auto fetchResult(Result* output) -> Result
{
    check(cudaGetLastError(), "kernel launch");
    check(cudaDeviceSynchronize(), "kernel execution");
    Result result{};
    check(cudaMemcpy(&result, output, sizeof(Result), cudaMemcpyDeviceToHost), "cudaMemcpy");
    check(cudaFree(output), "cudaFree");
    return result;
}

//======================================================================================================================
//
// KERNELS COMPILED WITH NVCC
//
//======================================================================================================================

/// Run kernel(args..., output) compiled by nvcc in a single device thread and return the result it writes to output.
/// The arguments must have exactly the types of the kernel parameters, since runWithNvrtc can not convert them.
template<typename Result, typename... Params, typename... Args>
auto runOnDevice(void (*kernel)(Params...), const Args&... args) -> Result
{
    static_assert(std::is_same_v<std::tuple<Params...>, std::tuple<Args..., Result*>>,
        "the arguments must have the types of the kernel parameters, the last one being the output pointer");
    initDevice();
    Result* output = nullptr;
    check(cudaMalloc(&output, sizeof(Result)), "cudaMalloc");
    kernel<<<1, 1>>>(args..., output);
    return fetchResult(output);
}

//======================================================================================================================
//
// KERNELS COMPILED WITH NVRTC
//
//======================================================================================================================

/// Return the code of an example file located between the lines AUTODIFF_NVRTC_BEGIN_SOURCE and AUTODIFF_NVRTC_END_SOURCE,
/// preceded by a #line directive so that NVRTC diagnostics refer to the lines of the example file.
inline auto extractNvrtcSource(const std::string& filename) -> std::string
{
    std::ifstream file(filename);
    if(!file)
        fail("opening " + filename, "can not read the file");

    std::stringstream source;
    std::string line;
    std::size_t lineno = 0;
    bool begin = false, end = false;
    while(!begin && std::getline(file, line)) {
        ++lineno;
        begin = line.rfind("AUTODIFF_NVRTC_BEGIN_SOURCE", 0) == 0;
    }
    source << "#line " << lineno + 1 << " \"" << filename << "\"\n";
    while(!end && std::getline(file, line)) {
        end = line.rfind("AUTODIFF_NVRTC_END_SOURCE", 0) == 0;
        if(!end)
            source << line << '\n';
    }
    if(!begin || !end)
        fail("extracting the NVRTC source of " + filename, "AUTODIFF_NVRTC_BEGIN_SOURCE or AUTODIFF_NVRTC_END_SOURCE not found");
    return source.str();
}

/// A kernel compiled at runtime with NVRTC and loaded in the current context.
struct NvrtcKernel
{
    CUmodule module = nullptr;
    CUfunction function = nullptr;
};

/// Compile with NVRTC the NVRTC source of an example file (see extractNvrtcSource) for the architecture of the current
/// device and load the kernel of the given name (a C++ name expression, e.g. "kernel" or "kernel<double>").
inline auto compileWithNvrtc(const std::string& filename, const std::string& kernelName) -> NvrtcKernel
{
    initDevice();

    int device = 0, major = 0, minor = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "cudaDeviceGetAttribute");
    check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "cudaDeviceGetAttribute");

    const std::string source = extractNvrtcSource(filename);
    const std::vector<std::string> options = {
        "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor),
        "-std=c++17",
        "-default-device", // required by Eigen and autodiff with NVRTC
        "-I" AUTODIFF_EXAMPLES_AUTODIFF_INCLUDE_DIR,
        "-I" AUTODIFF_EXAMPLES_EIGEN_INCLUDE_DIR,
        "-I" AUTODIFF_EXAMPLES_CUDA_INCLUDE_DIR,
        "-DAUTODIFF_EIGEN_FOUND",
    };
    std::vector<const char*> opts;
    for(const auto& option : options)
        opts.push_back(option.c_str());

    nvrtcProgram program;
    check(nvrtcCreateProgram(&program, source.c_str(), filename.c_str(), 0, nullptr, nullptr), "nvrtcCreateProgram");
    check(nvrtcAddNameExpression(program, kernelName.c_str()), "nvrtcAddNameExpression");

    const nvrtcResult res = nvrtcCompileProgram(program, int(opts.size()), opts.data());
    if(res != NVRTC_SUCCESS) {
        std::size_t logSize = 0;
        check(nvrtcGetProgramLogSize(program, &logSize), "nvrtcGetProgramLogSize");
        std::string log(logSize, '\0');
        check(nvrtcGetProgramLog(program, log.data()), "nvrtcGetProgramLog");
        std::cerr << "NVRTC compilation log of " << filename << ":\n" << log.c_str() << std::endl;
        check(res, "nvrtcCompileProgram");
    }

    const char* loweredName = nullptr;
    check(nvrtcGetLoweredName(program, kernelName.c_str(), &loweredName), "nvrtcGetLoweredName");
    const std::string mangledName = loweredName; // owned by the program

    std::size_t ptxSize = 0;
    check(nvrtcGetPTXSize(program, &ptxSize), "nvrtcGetPTXSize");
    std::string ptx(ptxSize, '\0');
    check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
    check(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");

    NvrtcKernel kernel;
    check(cuModuleLoadData(&kernel.module, ptx.c_str()), "cuModuleLoadData");
    check(cuModuleGetFunction(&kernel.function, kernel.module, mangledName.c_str()), "cuModuleGetFunction");
    return kernel;
}

/// Compile with NVRTC the NVRTC source of an example file, run kernel(args..., output) in a single device thread and
/// return the result it writes to output. The arguments must have exactly the types of the kernel parameters.
template<typename Result, typename... Args>
auto runWithNvrtc(const std::string& filename, const std::string& kernelName, const Args&... args) -> Result
{
    const NvrtcKernel kernel = compileWithNvrtc(filename, kernelName);
    Result* output = nullptr;
    check(cudaMalloc(&output, sizeof(Result)), "cudaMalloc");
    void* params[] = {const_cast<void*>(static_cast<const void*>(&args))..., &output};
    check(cuLaunchKernel(kernel.function, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr), "cuLaunchKernel");
    const Result result = fetchResult(output);
    check(cuModuleUnload(kernel.module), "cuModuleUnload");
    return result;
}

//======================================================================================================================
//
// COMPARISON OF THE RESULTS
//
//======================================================================================================================

/// The relative tolerance used to compare results computed on the host and on the device (the device math functions
/// may differ from the host ones by a few ulps).
constexpr double rtol = 1e-12;

/// The absolute tolerance used to compare results close to zero.
constexpr double atol = 1e-12;

inline bool allclose(double x, double y)
{
    return std::abs(x - y) <= atol + rtol * std::max(std::abs(x), std::abs(y));
}

template<typename DerivedX, typename DerivedY>
bool allclose(const Eigen::DenseBase<DerivedX>& x, const Eigen::DenseBase<DerivedY>& y)
{
    if(x.rows() != y.rows() || x.cols() != y.cols())
        return false;
    for(Eigen::Index i = 0; i < x.rows(); ++i)
        for(Eigen::Index j = 0; j < x.cols(); ++j)
            if(!allclose(x(i, j), y(i, j)))
                return false;
    return true;
}

template<typename T, std::size_t N>
bool allclose(const std::array<T, N>& x, const std::array<T, N>& y)
{
    for(std::size_t i = 0; i < N; ++i)
        if(!allclose(x[i], y[i]))
            return false;
    return true;
}

//======================================================================================================================
//
// OUTPUT
//
//======================================================================================================================

inline void printBanner()
{
    std::cout << std::string(100, '*') << std::endl;
}

/// Print the name of the example being run.
inline void printHeader(const std::string& filename)
{
    printBanner();
    std::cout << "Running example " << filename.substr(filename.find_last_of("/\\") + 1) << std::endl;
    std::cout << std::endl;
}

/// Compare the results computed on the host, on the device with nvcc and on the device with NVRTC with the given
/// predicate, print the outcome and return the exit code of the example.
template<typename Results, typename Predicate>
int exitExample(const Results& host, const Results& device, const Results& nvrtc, Predicate close)
{
    const bool hostDevice = close(host, device);
    const bool hostNvrtc = close(host, nvrtc);
    const bool deviceNvrtc = close(device, nvrtc);
    const bool success = hostDevice && hostNvrtc && deviceNvrtc;
    if(success)
        std::cout << "SUCCESS, host, device and NVRTC results match." << std::endl;
    else {
        const char* cmp[2] = {"!=", "=="};
        std::cerr << "Host, device and NVRTC results do not match:" << std::endl;
        std::cerr << "    host results " << cmp[hostDevice] << " device results" << std::endl;
        std::cerr << "    host results " << cmp[hostNvrtc] << " NVRTC results" << std::endl;
        std::cerr << "  device results " << cmp[deviceNvrtc] << " NVRTC results" << std::endl;
        std::cerr << "FAILURE, this is a bug, please open an issue." << std::endl;
    }
    printBanner();
    std::cout << std::endl;
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

/// Compare the results computed on the host, on the device with nvcc and on the device with NVRTC with allclose.
template<typename Results>
int exitExample(const Results& host, const Results& device, const Results& nvrtc)
{
    return exitExample(host, device, nvrtc, [](const Results& x, const Results& y) { return allclose(x, y); });
}

} // namespace examples
} // namespace autodiff
