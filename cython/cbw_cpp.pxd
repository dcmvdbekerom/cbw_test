cdef extern from "crs_cpp.h":

    #import <cuda.h>
    void initCUDA();
    void finalizeCUDA();
    void setupDeviceMemory(CUdeviceptr* d_a, CUdeviceptr* d_b, CUdeviceptr* d_c);
    void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);
    void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);
