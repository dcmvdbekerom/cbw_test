from ctypes import *
import sys

#TODO: make compatible with linux/mac
#TODO: implement cpu compatibility mode
#TODO: establish cuda version requirement

lib = windll.LoadLibrary('nvcuda.dll')

CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

#TODO: complete list, get rid of _v2's 
cuDeviceGetCount = lib.cuDeviceGetCount
cuDeviceGet = lib.cuDeviceGet

class cuContext:
    
    def __init__(self, device_id=0, flags=0):
        err = lib.cuInit(0)

        deviceCount = c_int(0)
        lib.cuDeviceGetCount(byref(deviceCount))

        if (deviceCount == 0):
            print("Error: no devices supporting CUDA\n")
            sys.exit()

        self.device = c_long(0)
        lib.cuDeviceGet(byref(self.device), device_id)
        
        self.context = c_longlong(0)
        err = lib.cuCtxCreate_v2(byref(self.context), flags, self.device)
        if (err != CUDA_SUCCESS):
            print("* Error initializing the CUDA context.")
            lib.cuCtxDestroy_v2(self.context)
            sys.exit()
            
    
    @staticmethod
    def getDeviceList():
        dev_list = []
        deviceCount = c_int(0)

        lib.cuInit(0)
        lib.cuDeviceGetCount(byref(deviceCount))

        for i in range(deviceCount.value):

            device = c_long(0)
            lib.cuDeviceGet(byref(device), i)
       
            name = create_string_buffer(100)        
            lib.cuDeviceGetName(name, 100, device)
            dev_list.append(name.value.decode())
        
        return dev_list
    
    
    def printDeviceCapabilities(self):
        
        major = c_long(0)
        minor = c_long(0)  
        lib.cuDeviceGetAttribute(byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.device)
        lib.cuDeviceGetAttribute(byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.device)        
        print("> GPU Device has SM {:d}.{:d} compute capability".format(major.value, minor.value))
        
        totalGlobalMem = c_size_t(0)
        lib.cuDeviceTotalMem_v2(byref(totalGlobalMem), self.device)
        print("  Total amount of global memory:   {:d} bytes".format(totalGlobalMem.value))
        print("  64-bit Memory Address:           {:s}".format("YES" if totalGlobalMem.value > (2<<31) else "NO"))

    def synchronize(self):
        self.context.cuCtxSynchronize()
    
    def destroy(self):
        lib.cuCtxDestroy_v2(self.context);

    def __del__(self):
        self.destry()
        print('Context destroyed')


class cuArray:

    def __init__(self, arr):
        self.arr_h = arr
        self.dtype = arr.dtype
        self.size = arr.size
        self.itemsize = arr.dtype.itemsize
        self.bytesize = arr.itemsize * arr.size

        self.arr_d = c_void_p()        
        lib.cuMemAlloc_v2(byref(self.arr_d), self.bytesize)
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

        lib.cuMemcpyHtoD_v2(self.arr_d, c_void_p(self.arr_h.ctypes.data), self.bytesize)

    def d2h(self):

        lib.cuMemcpyDtoH_v2(c_void_p(self.arr_h.ctypes.data), self.arr_d, self.bytesize)
        return self.arr_h

    def __del__(self):

        lib.cuMemFree_v2(self.arr_d)

      
class cuFunction:
    def __init__(self, fptr):
        self.fptr = fptr
        self.retvars = None
    
    def set_grid(self, blocks=(1,1,1), threads=(1,1,1)):
        self.blocks = blocks
        self.threads = threads
    
    def set_retvars(self, retvars):
        self.retvars = retvars
        
    def __call__(self, *vargs, **kwargs):
    
        try:
            self.blocks = kwargs['blocks']
        except(KeyError):
            pass
            
        try:
            self.threads = kwargs['threads']
        except(KeyError):
            pass

        try:
            self.retvars = kwargs['retvars']
        except(KeyError):
            pass

        if self.retvars is None:
            self.retvars = len(vargs)*[False]
            self.retvars[-1] = True
            
        voidPtrArr = len(vargs)*c_void_p 
        cargs = voidPtrArr(*[cast(byref(arr.arr_d), c_void_p) for arr in vargs])
                           
        lib.cuLaunchKernel(self.fptr, *self.blocks, *self.threads, 0, 0, cargs, 0)
        
        for arr, is_retvar in zip(vargs, self.retvars):
            if is_retvar:
                arr.is_uptodate = False


class cuModule:
    def __init__(self, context, module_name):
        self.module_name = module_name
        module_file = c_char_p(module_name.encode())        
        self.context_obj = context
        self.module = c_longlong(0)
        err = lib.cuModuleLoad(byref(self.module), module_file)
        if (err != CUDA_SUCCESS):
            print("* Error loading the module {:s}\n".format(module_file.value.decode()))
            self.context_obj.destroy()
            sys.exit() 
            
        self.func_dict = {}

    def __getattr__(self, attr):
        try:
            self.func_dict[attr]
            return self.func_dict[attr]
            
        except(KeyError):
            function = c_longlong(0)
            kernel_name = c_char_p(attr.encode())
            err = lib.cuModuleGetFunction(byref(function), self.module, kernel_name)
            if (err != CUDA_SUCCESS):
                print("* Error getting kernel function {:s}".format(kernel_name.value.decode()))
                self.context_obj.destroy()
                sys.exit()
                
            self.func_dict[attr] = cuFunction(function)
            return self.func_dict[attr] 

    def setConstant(self, name, c_val):
        var = c_void_p()
        size = c_long()

        lib.cuModuleGetGlobal_v2(byref(var), byref(size), self.module, c_char_p(name.encode()))
        lib.cuMemcpyHtoD_v2(var, byref(c_val), size)
        


