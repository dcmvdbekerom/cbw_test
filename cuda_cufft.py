from ctypes import *
import nvidia.cufft
cufft_path = nvidia.cufft.__path__[0] + '\\bin\\'

lib = windll.LoadLibrary(cufft_path + 'cufft64_11.dll')


plan = c_longlong(0)


print(lib.cufftCreate(byref(plan)))

#CUFFT_CALL(cufftCreate(&plan));
#CUFFT_CALL(cufftPlan1d(&plan, input.size(), CUFFT_R2C, batch_size));
