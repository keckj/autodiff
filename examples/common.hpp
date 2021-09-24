//                  _  _
//  _   _|_ _  _|o_|__|_
// (_||_||_(_)(_|| |  |
//
// automatic differentiation made easier in C++
// https://github.com/autodiff/autodiff
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2020 Allan Leal
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

#include <limits>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <iostream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include "nvrtc.h"
#include "cuda.h"

#include <autodiff/common/eigen.hpp>

#define XSTR(s) STR(s)
#define STR(s) #s

#define AUTODIFF_NVRTC_BEGIN_SOURCE
#define AUTODIFF_NVRTC_END_SOURCE

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

template<typename T, size_t N>
    __host__ bool allclose(const cuda::std::array<T,N>& x, const cuda::std::array<T,N>& y, size_t n=1) {
        for (size_t j = 0; j < N; j++) {
            if (!allclose(x[j], y[j], n)) {
                return false;
            }
        }
        return true;
    }


__host__ void print_banner() {
    for (size_t i = 0; i < 100; ++i) {
        std::cout << '*'; 
    }
    std::cout << std::endl;
}

__host__ void print_example_header(const std::string& filename) {
    print_banner();
    std::cout << "Running example " << filename << std::endl;
    std::cout << std::endl;
}


template <typename T>
    [[nodiscard]] 
    __host__ int exit_example(
            const T& host_results, const T& device_results, const T& nvrtc_results,
            const std::function<bool(const T&, const T&)>& fn = [](const T& lhs, const T& rhs) -> bool { return allclose(lhs, rhs); }) {
        bool host_and_device_match  = fn(host_results, device_results);
        bool host_and_nvrtc_match   = fn(host_results, nvrtc_results);
        bool device_and_nvrtc_match = fn(device_results, nvrtc_results);
        bool success = host_and_device_match && host_and_nvrtc_match && device_and_nvrtc_match;
        if (success) {
            std::cout << std::endl;
            std::cout << "SUCCESS, all results match." << std::endl;
        }
        else {
            static const std::string cmp[2] = {"!=", "=="};
            std::cerr << std::endl;
            std::cerr << "Host, device and nvrtc results do not match:" << std::endl;
            std::cerr << "    host results " << cmp[host_and_device_match]  << " device results" << std::endl;
            std::cerr << "    host results " << cmp[host_and_nvrtc_match]   << " nvrtc results" << std::endl;
            std::cerr << "  device results " << cmp[device_and_nvrtc_match] << " nvrtc results" << std::endl;
            std::cerr << "FAILURE, this is a bug, please open an issue." << std::endl;
        }
        print_banner();
        std::cout << std::endl;
        return success ? EXIT_SUCCESS : EXIT_FAILURE;
    }


[[noreturn]] void nvrtc_fail(const std::string& msg, nvrtcResult code) {
    const char *error_string = nvrtcGetErrorString(code);
    std::cerr << "error: " << msg << " (" << code << ')' << std::endl;
    std::cerr << error_string << std::endl;
    std::exit(EXIT_FAILURE);
}

[[noreturn]] void cuda_fail(const std::string& msg, CUresult code) {
    const char *error_name, *error_string;
    cuGetErrorName(code, &error_name);
    cuGetErrorString(code, &error_string);
    std::cerr << "error: " << msg << " (" << code << ')' << std::endl;
    std::cerr << error_name  << ": " << error_string << std::endl;
    std::exit(EXIT_FAILURE);
}


CUfunction build_nvrtc_kernel(const std::string& filename, const char* kernel_name = "kernel") {
    /* extract kernel source from the example file */
    std::string source;
    {
        std::stringstream source_stream;
        std::fstream fs;
        bool start_found(false), end_found(false);

        fs.open(filename, std::fstream::in);
        {
            std::string line;
            while (std::getline(fs, line)) {
                if (line.rfind("AUTODIFF_NVRTC_BEGIN_SOURCE", 0) == 0) {
                    start_found = true;
                    break;
                }
            }
            while (std::getline(fs, line)) {
                if (line.rfind("AUTODIFF_NVRTC_END_SOURCE", 0) == 0) {
                    end_found = true;
                    break;
                }
                source_stream << line << std::endl;
            }
        }
        fs.close();

        if (!start_found || !end_found) {
            throw std::invalid_argument("Invalid nvrtc source file '" + filename + "'.");
        }
        source = source_stream.str();
    }

    /* build the kernel */
    nvrtcProgram prog;
    const char* mangled_kernel_name;
    {
        nvrtcResult rv;
        rv = nvrtcCreateProgram(
                &prog,
                source.c_str(),
                "kernel.cu",
                0,
                nullptr,
                nullptr
                );
        if(rv != NVRTC_SUCCESS) nvrtc_fail("nvrtcCreateProgram", rv);

        rv = nvrtcAddNameExpression(prog, kernel_name);
        if(rv != NVRTC_SUCCESS) nvrtc_fail("nvrtcAddNameExpression", rv);

        std::vector<const char*> options = {
            "-std=c++17",
            "-default-device",
            "--gpu-architecture", "compute_" XSTR(AUTODIFF_CUDA_ARCHITECTURE),
            "-I" XSTR(AUTODIFF_PROJECT_ROOT),
            "-I" XSTR(AUTODIFF_CUDA_INCLUDE_DIR),
            "-DAUTODIFF_USE_INTERNAL_EIGEN",
        };

        rv = nvrtcCompileProgram(prog, options.size(), options.data());
        if(rv != NVRTC_SUCCESS) {
            std::size_t log_size;
            rv = nvrtcGetProgramLogSize(prog, &log_size);
            if(rv != NVRTC_SUCCESS) nvrtc_fail("nvrtcGetProgramLogSize", rv);

            auto log = std::make_unique<char[]>(log_size);
            rv = nvrtcGetProgramLog(prog, log.get());
            if(rv != NVRTC_SUCCESS) nvrtc_fail("nvrtcGetProgramLog", rv);
            assert(log[log_size - 1] == '\0');

            std::cerr << "Compile error; log:\n" << log.get() << std::endl;

            nvrtc_fail("nvrtcCompileProgram", rv);
        }

        rv = nvrtcGetLoweredName(prog, kernel_name, &mangled_kernel_name);
        if(rv != NVRTC_SUCCESS) nvrtc_fail("nvrtcGetLoweredName", rv);
    }
    
    /* Extract PTX and destroy program */
    std::unique_ptr<char[]> ptx;
    {
        nvrtcResult rv;
        std::size_t ptx_size;

        rv = nvrtcGetPTXSize(prog, &ptx_size);
        if(rv != NVRTC_SUCCESS) nvrtc_fail("nvrtcGetPTXSize", rv);

        ptx = std::make_unique<char[]>(ptx_size);
        rv = nvrtcGetPTX(prog, ptx.get());
        
        if(rv != NVRTC_SUCCESS) nvrtc_fail("nvrtcGetPTX", rv);
        assert(ptx[ptx_size - 1] == '\0');
    }

    /* Build module and extract function */
    CUfunction kernel;
    {
        CUresult rv;
        CUmodule cuda_module;

        rv = cuModuleLoadData(&cuda_module, ptx.get());
        if(rv != CUDA_SUCCESS) cuda_fail("cuModuleLoadData", rv);

        rv = cuModuleGetFunction(&kernel, cuda_module, mangled_kernel_name);
        if(rv != CUDA_SUCCESS) cuda_fail("cuModuleGetFunction", rv);
        
        nvrtcDestroyProgram(&prog);
    }

    return kernel;
}

} /* end of namespace autodiff */
