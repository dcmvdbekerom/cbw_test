import numpy as np
from ctypes_cuda import cuArr, initCUDA, runKernel, finalizeCUDA


print("- Initializing...");
funlist = initCUDA("cu/matSumKernel.ptx", ["matSum"])

N = 100

dtype = np.int32
a = cuArr(N - np.arange(N, dtype=dtype))
b = cuArr(np.arange(N, dtype=dtype)**2)
c = cuArr(np.zeros(N, dtype=dtype), is_returnvar=True)


print("# Running the kernel...")
#grid for kernel: <<<N, 1>>>
blocks = (N,1,1)
threads = (1,1,1)  
runKernel(funlist["matSum"], blocks, threads, [a, b, c])
print("# Kernel complete.")


for i in range(N):
    if (c[i] != a[i] + b[i]):
        print("* Error at array position {:d}: Expected {:d}, Got {:d}".format(
            i, a[i] + b[i], c[i]))
        
print("*** All checks complete.")


print("- Finalizing...");
finalizeCUDA()
