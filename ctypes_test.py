from ctypes import *
import numpy as np
import sys

mod = windll.LoadLibrary('nvcuda.dll')
##
## // This will output the proper CUDA error strings
## // in the event that a CUDA host call returns an error
###define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
##
##inline void __checkCudaErrors(CUresult err, const char* file, const int line)
##{
##    if (CUDA_SUCCESS != err) {
##        fprintf(stderr,
##            "CUDA Driver API error = %04d from file <%s>, line %i.\n",
##            err, file, line);
##        exit(-1);
##    }
##}
##
##// --- global variables ----------------------------------------------------

N = 100
sizeof_int = 4

device = c_long(0)
context = c_longlong(0)
module = c_longlong(0)
function = c_longlong(0)
totalGlobalMem = c_size_t(0)

module_file = c_char_p(b"cu/matSumKernel.ptx")
kernel_name = c_char_p(b"matSum")

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

cuDeviceGetCount = mod.cuDeviceGetCount
cuDeviceGet = mod.cuDeviceGet


def initCUDA():

    deviceCount = 0
    err = mod.cuInit(0)
    major = c_long(0)
    minor = c_long(0)

    deviceCount = c_int(0)
    mod.cuDeviceGetCount(byref(deviceCount))

    if (deviceCount == 0):
        print("Error: no devices supporting CUDA\n")
        sys.exit()
 
    #get first CUDA device
    mod.cuDeviceGet(byref(device), 0)
    name = create_string_buffer(100) 
    mod.cuDeviceGetName(name, 100, device)
    print("> Using device 0: {:s}".format(name.value.decode()))

    #get compute capabilities and the devicename
    mod.cuDeviceGetAttribute(byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    mod.cuDeviceGetAttribute(byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)

    print("> GPU Device has SM {:d}.{:d} compute capability".format(major.value, minor.value))

    mod.cuDeviceTotalMem_v2(byref(totalGlobalMem), device)
    print("  Total amount of global memory:   {:d} bytes".format(totalGlobalMem.value))
    print("  64-bit Memory Address:           {:s}".format("YES" if totalGlobalMem.value > (2<<31) else "NO"))
    
    err = mod.cuCtxCreate_v2(byref(context), 0, device)
    if (err != CUDA_SUCCESS):
        print("* Error initializing the CUDA context.")
        mod.cuCtxDestroy_v2(context)
        sys.exit()
        
    err = mod.cuModuleLoad(byref(module), module_file)
    if (err != CUDA_SUCCESS):
        print(err)
        print("* Error loading the module {:s}\n".format(module_file.value.decode()))
        mod.cuCtxDestroy_v2(context)
        sys.exit()

    err = mod.cuModuleGetFunction(byref(function), module, kernel_name)
    if (err != CUDA_SUCCESS):
        print(err)
        print("* Error getting kernel function {:s}".format(kernel_name.value.decode()))
        mod.cuCtxDestroy_v2(context)
        sys.exit()

    
def finalizeCUDA():

    #//cuCtxDetach(context);
    mod.cuCtxDestroy_v2(context);


def setupDeviceMemory(d_a, d_b, d_c):

    mod.cuMemAlloc_v2(byref(d_a), sizeof_int * N)
    mod.cuMemAlloc_v2(byref(d_b), sizeof_int * N)
    mod.cuMemAlloc_v2(byref(d_c), sizeof_int * N)


def releaseDeviceMemory(d_a, d_b, d_c):
    
    mod.cuMemFree_v2(byref(d_a))
    mod.cuMemFree_v2(d_b)
    mod.cuMemFree_v2(d_c)


def runKernel(d_a, d_b, d_c):

    #void* args[3] = { &d_a, &d_b, &d_c };
    threeVoidPtrs = 3*c_void_p 
    args = threeVoidPtrs(cast(byref(d_a), c_void_p),
                         cast(byref(d_b), c_void_p),
                         cast(byref(d_c), c_void_p))

    #grid for kernel: <<<N, 1>>>
    mod.cuLaunchKernel_ptsz(function, N, 1, 1,  #ptsz  // Nx1x1 blocks
        1, 1, 1,            #// 1x1x1 threads
        0, 0, args, 0)

a = N - np.arange(N, dtype=np.int32)
b = np.arange(N, dtype=np.int32)**2
c = np.zeros(N, dtype=np.int32)


#CUdeviceptr d_a, d_b, d_c;
d_a = c_void_p()
d_b = c_void_p()
d_c = c_void_p()

#initialize
print("- Initializing...");
initCUDA();

#allocate memory
setupDeviceMemory(d_a, d_b, d_c)


#copy arrays to device
mod.cuMemcpyHtoD_v2(d_a, c_void_p(a.ctypes.data), sizeof_int * N)
mod.cuMemcpyHtoD_v2(d_b, c_void_p(b.ctypes.data), sizeof_int * N)

#run
print("# Running the kernel...")
runKernel(d_a, d_b, d_c)
print("# Kernel complete.")

#copy results to host and report
mod.cuMemcpyDtoH_v2(c_void_p(c.ctypes.data), d_c, sizeof_int * N)

for i in range(N):
    if (c[i] != a[i] + b[i]):
        print("* Error at array position {:d}: Expected {:d}, Got {:d}".format(
            i, a[i] + b[i], c[i]))
print("*** All checks complete.")

#finish
print("- Finalizing...");
releaseDeviceMemory(d_a, d_b, d_c);
finalizeCUDA()

