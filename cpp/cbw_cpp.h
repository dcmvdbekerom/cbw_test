/*
// --- global variables ----------------------------------------------------
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
size_t     totalGlobalMem;

char* module_file = (char*)"matSumKernel.ptx";
char* kernel_name = (char*)"matSum";
*/

// --- functions -----------------------------------------------------------

#import <cuda.h>

void initCUDA();
void finalizeCUDA();
void setupDeviceMemory(CUdeviceptr* d_a, CUdeviceptr* d_b, CUdeviceptr* d_c);
void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);
void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);
