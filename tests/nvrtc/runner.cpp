// autodiff-nvrtctests: compiles every test kernel file with NVRTC under several option sets and, when a CUDA device
// is available, runs every test kernel and reports the number of failed device-side checks.
//
// Usage: autodiff-nvrtctests [--compile-only] [--verbose] [filter...]
//   filter: only kernel files whose name contains one of the given substrings are processed.

#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define NVRTC_CHECK(call)                                                                              \
    do {                                                                                               \
        nvrtcResult res = call;                                                                        \
        if(res != NVRTC_SUCCESS) {                                                                     \
            std::fprintf(stderr, "%s failed: %s\n", #call, nvrtcGetErrorString(res));                  \
            std::exit(2);                                                                              \
        }                                                                                              \
    } while(0)

static bool cuCheck(CUresult res, const char* what)
{
    if(res == CUDA_SUCCESS)
        return true;
    const char* msg = nullptr;
    cuGetErrorString(res, &msg);
    std::fprintf(stderr, "%s failed: %s\n", what, msg ? msg : "unknown error");
    return false;
}

struct OptionSet
{
    std::string name;
    std::vector<std::string> options;
};

static std::string readFile(const fs::path& path)
{
    std::ifstream file(path);
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static std::vector<std::string> findTests(const std::string& source)
{
    std::vector<std::string> names;
    std::regex re(R"(^\s*AUTODIFF_NVRTC_TEST\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\))");
    std::istringstream lines(source);
    std::string line;
    std::smatch m;
    while(std::getline(lines, line))
        if(std::regex_search(line, m, re))
            names.push_back(m[1]);
    return names;
}

int main(int argc, char** argv)
{
    bool compileOnly = false;
    bool verbose = false;
    std::vector<std::string> filters;
    for(int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if(arg == "--compile-only")
            compileOnly = true;
        else if(arg == "--verbose")
            verbose = true;
        else
            filters.push_back(arg);
    }

    // Device setup (optional)
    CUdevice device = 0;
    CUcontext context = nullptr;
    int major = 7, minor = 5;
    if(!compileOnly) {
        int count = 0;
        if(cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) {
            std::printf("No CUDA device found: running in compile-only mode.\n");
            compileOnly = true;
        }
        else {
            cuCheck(cuDeviceGet(&device, 0), "cuDeviceGet");
            cuCheck(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "cuDeviceGetAttribute");
            cuCheck(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "cuDeviceGetAttribute");
            cuCheck(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
            cuCheck(cuCtxSetCurrent(context), "cuCtxSetCurrent");
            // Reverse mode allocates its expression tree on the device heap and recurses on the stack
            cuCheck(cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, size_t(256) << 20), "cuCtxSetLimit(heap)");
            cuCheck(cuCtxSetLimit(CU_LIMIT_STACK_SIZE, 16 * 1024), "cuCtxSetLimit(stack)");
        }
    }
    const std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);

    const std::vector<std::string> common = {
        arch,
        "-I" AUTODIFF_NVRTC_AUTODIFF_INCLUDE_DIR,
        "-I" AUTODIFF_NVRTC_EIGEN_INCLUDE_DIR,
        "-I" AUTODIFF_NVRTC_CUDA_INCLUDE_DIR,
        "-I" AUTODIFF_NVRTC_TESTS_DIR,
        "-DAUTODIFF_EIGEN_FOUND",
    };
    const std::vector<OptionSet> optionSets = {
        {"c++17", {"-std=c++17", "-default-device"}},
        {"c++20", {"-std=c++20", "-default-device"}},
    };

    std::vector<fs::path> files;
    for(const auto& entry : fs::directory_iterator(AUTODIFF_NVRTC_KERNELS_DIR))
        if(entry.path().extension() == ".cu")
            files.push_back(entry.path());
    std::sort(files.begin(), files.end());

    int nfailedCompilations = 0, nfailedTests = 0, npassedTests = 0;

    for(const auto& path : files) {
        const std::string filename = path.filename().string();
        if(!filters.empty() && std::none_of(filters.begin(), filters.end(), [&](auto& f) { return filename.find(f) != std::string::npos; }))
            continue;

        const std::string source = readFile(path);
        const auto tests = findTests(source);

        for(const auto& optset : optionSets) {
            nvrtcProgram prog;
            NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), filename.c_str(), 0, nullptr, nullptr));
            for(const auto& test : tests)
                NVRTC_CHECK(nvrtcAddNameExpression(prog, test.c_str()));

            std::vector<const char*> opts;
            for(const auto& o : common)
                opts.push_back(o.c_str());
            for(const auto& o : optset.options)
                opts.push_back(o.c_str());

            const nvrtcResult res = nvrtcCompileProgram(prog, int(opts.size()), opts.data());
            size_t logSize = 0;
            nvrtcGetProgramLogSize(prog, &logSize);
            std::string log(logSize, '\0');
            nvrtcGetProgramLog(prog, log.data());

            if(res != NVRTC_SUCCESS) {
                ++nfailedCompilations;
                std::printf("[COMPILE FAIL] %s (%s)\n%s\n", filename.c_str(), optset.name.c_str(), log.c_str());
                nvrtcDestroyProgram(&prog);
                continue;
            }
            std::printf("[COMPILE OK]   %s (%s)\n", filename.c_str(), optset.name.c_str());
            if(verbose && logSize > 1)
                std::printf("%s\n", log.c_str());

            if(!compileOnly) {
                size_t ptxSize = 0;
                NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
                std::string ptx(ptxSize, '\0');
                NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

                CUmodule module;
                if(!cuCheck(cuModuleLoadData(&module, ptx.c_str()), "cuModuleLoadData")) {
                    ++nfailedTests;
                    nvrtcDestroyProgram(&prog);
                    continue;
                }

                CUdeviceptr dnfailures;
                cuCheck(cuMemAlloc(&dnfailures, sizeof(unsigned int)), "cuMemAlloc");

                for(const auto& test : tests) {
                    const char* lowered = nullptr;
                    NVRTC_CHECK(nvrtcGetLoweredName(prog, test.c_str(), &lowered));
                    CUfunction kernel;
                    cuCheck(cuModuleGetFunction(&kernel, module, lowered), "cuModuleGetFunction");
                    cuCheck(cuMemsetD32(dnfailures, 0, 1), "cuMemsetD32");
                    void* args[] = {&dnfailures};
                    const unsigned nthreads = 32;
                    bool ok = cuCheck(cuLaunchKernel(kernel, 1, 1, 1, nthreads, 1, 1, 0, nullptr, args, nullptr), "cuLaunchKernel");
                    ok = ok && cuCheck(cuCtxSynchronize(), ("running " + test).c_str());
                    unsigned int nfailures = 0;
                    if(ok)
                        cuCheck(cuMemcpyDtoH(&nfailures, dnfailures, sizeof(unsigned int)), "cuMemcpyDtoH");
                    if(!ok || nfailures) {
                        ++nfailedTests;
                        std::printf("  [FAIL] %s: %u failed checks%s\n", test.c_str(), nfailures, ok ? "" : " (launch error)");
                        if(!ok)
                            return 1; // the context is likely corrupted
                    }
                    else {
                        ++npassedTests;
                        if(verbose)
                            std::printf("  [PASS] %s\n", test.c_str());
                    }
                }
                cuMemFree(dnfailures);
                cuModuleUnload(module);
            }
            nvrtcDestroyProgram(&prog);
        }
    }

    std::printf("\n%d compilation failure(s), %d test(s) passed, %d test(s) failed%s\n", nfailedCompilations, npassedTests,
        nfailedTests, compileOnly ? " (compile-only mode)" : "");
    if(context)
        cuDevicePrimaryCtxRelease(device);
    return (nfailedCompilations || nfailedTests) ? 1 : 0;
}
