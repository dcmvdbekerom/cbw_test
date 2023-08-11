cdef extern from "../cpp/crs_cpp.h":

    #import <cuda.h>
    void initCUDA();
    void finalizeCUDA();
    void setupDeviceMemory(CUdeviceptr* d_a, CUdeviceptr* d_b, CUdeviceptr* d_c);
    void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);
    void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);

cdef extern from "../cpp/CUDA/v11.6/include/cuda.h":
    int cuMemcpyHtoD(CUdeviceptr dstDevice, cons void* srcHost, size_t ByteCount)
    int cuMemcpyDtoH(cons void* dstHost, CUdeviceptr srcDevice, size_t ByteCount)