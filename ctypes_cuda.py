from ctypes import *
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

device = c_long(0)
context = c_longlong(0)
module = c_longlong(0)
totalGlobalMem = c_size_t(0)
_numDeviceAllocs = 0

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

cuDeviceGetCount = mod.cuDeviceGetCount
cuDeviceGet = mod.cuDeviceGet

class cuArr:
    def __init__(self, arr, is_returnvar=False):
        global _numDeviceAllocs
        self.arr_h = arr
        self.is_returnvar = is_returnvar
        self.dtype = arr.dtype
        self.size = arr.size
        self.itemsize = arr.dtype.itemsize
        self.bytesize = arr.itemsize * arr.size

        self.arr_d = c_void_p()
        mod.cuMemAlloc_v2(byref(self.arr_d), self.bytesize)
        _numDeviceAllocs += 1
        self.h2d()
        self.is_uptodate = True

    def __getitem__(self, i):
        if not self.is_uptodate:
            self.d2h()
            self.is_uptodate = True
        return self.arr_h[i]

    def __setitem__(self, i, a):

        self.arr_h[i] = a
        self.h2d()

    def h2d(self):
        mod.cuMemcpyHtoD_v2(self.arr_d, c_void_p(self.arr_h.ctypes.data), self.bytesize)

    def d2h(self):
        mod.cuMemcpyDtoH_v2(c_void_p(self.arr_h.ctypes.data), self.arr_d, self.bytesize)
        return self.arr_h

    def __del__(self):
        mod.cuMemFree_v2(self.arr_d)
        print('Deleted ',self)

        


def initCUDA(module_name, function_list):

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

    module_file = c_char_p(module_name.encode())        
    err = mod.cuModuleLoad(byref(module), module_file)
    if (err != CUDA_SUCCESS):
        print(err)
        print("* Error loading the module {:s}\n".format(module_file.value.decode()))
        mod.cuCtxDestroy_v2(context)
        sys.exit()

    ret = {}
    function = c_longlong(0)

    for f in function_list:
        kernel_name = c_char_p(f.encode())
        err = mod.cuModuleGetFunction(byref(function), module, kernel_name)
        if (err != CUDA_SUCCESS):
            print(err)
            print("* Error getting kernel function {:s}".format(kernel_name.value.decode()))
            mod.cuCtxDestroy_v2(context)
            sys.exit()
        ret[f] = function
    return ret


def finalizeCUDA():

    mod.cuCtxDestroy_v2(context);


def runKernel(function, blocks, threads, arg_list):

    voidPtrArr = len(arg_list)*c_void_p 
    args = voidPtrArr(*[cast(byref(arr.arr_d), c_void_p) for arr in arg_list])
                       
    mod.cuLaunchKernel(function, *blocks,  #ptsz 
        *threads,         
        0, 0, args, 0)
    
    for arr in arg_list:
        if arr.is_returnvar:
            arr.is_uptodate = False


