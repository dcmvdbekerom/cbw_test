/*
 * drivertest.cpp
 * Vector addition (host code)
 *
 * Andrei de A. Formiga, 2012-06-04
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

#include "../cu/matSumKernel.h"

 // This will output the proper CUDA error strings
 // in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char* file, const int line)
{
    if (CUDA_SUCCESS != err) {
        fprintf(stderr,
            "CUDA Driver API error = %04d from file <%s>, line %i.\n",
            err, file, line);
        exit(-1);
    }
}

// --- global variables ----------------------------------------------------
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
size_t     totalGlobalMem;

char* module_file = (char*)"../cu/matSumKernel.ptx";
char* kernel_name = (char*)"matSum";


// --- functions -----------------------------------------------------------
void initCUDA()
{
    int deviceCount = 0;
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s\n", name);

    // get compute capabilities and the devicename
    //checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device));
    printf("  Total amount of global memory:   %llu bytes\n",
        (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:           %s\n",
        (totalGlobalMem > (unsigned long long)4 * 1024 * 1024 * 1024L) ?
        "YES" : "NO");

    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        //cuCtxDetach(context);
        cuCtxDestroy(context);
        exit(-1);
    }

    err = cuModuleLoad(&module, module_file);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s\n", module_file);
        //cuCtxDetach(context);
        cuCtxDestroy(context);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        //cuCtxDetach(context);
        cuCtxDestroy(context);
        exit(-1);
    }
}

void finalizeCUDA()
{
    //cuCtxDetach(context);
    cuCtxDestroy(context);
}

void setupDeviceMemory(CUdeviceptr* d_a, CUdeviceptr* d_b, CUdeviceptr* d_c)
{
    checkCudaErrors(cuMemAlloc(d_a, sizeof(int) * N));
    checkCudaErrors(cuMemAlloc(d_b, sizeof(int) * N));
    checkCudaErrors(cuMemAlloc(d_c, sizeof(int) * N));
}

void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
{
    checkCudaErrors(cuMemFree(d_a));
    checkCudaErrors(cuMemFree(d_b));
    checkCudaErrors(cuMemFree(d_c));
}

void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c)
{
    void* args[3] = { &d_a, &d_b, &d_c };

    // grid for kernel: <<<N, 1>>>
    checkCudaErrors(cuLaunchKernel(function, N, 1, 1,  // Nx1x1 blocks
        1, 1, 1,            // 1x1x1 threads
        0, 0, args, 0));
}