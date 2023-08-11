import numpy as np
from ctypes_cuda import cuContext, cuArray, cuModule

N = 100
dtype = np.int32

print("- Initializing...");
for i, dev in enumerate(cuContext.getDeviceList()):
    print('Device {:d}: {:s}'.format(i,dev))
ctx = cuContext()
ctx.printDeviceCapabilities()

a = cuArray(N - np.arange(N, dtype=dtype))
b = cuArray(np.arange(N, dtype=dtype)**2)
c = cuArray(np.zeros(N, dtype=dtype))

mod = cuModule(ctx, "cu/matSumKernel.ptx")
mod.matSum.set_grid(blocks=(N,1,1))
mod.matSum.set_retvars([False, False, True])

print("# Running the kernel...")
mod.matSum(a,b,c)
print("# Kernel complete.")

for i in range(N):
    if (c[i] != a[i] + b[i]):
        print("* Error at array position {:d}: Expected {:d}, Got {:d}".format(
            i, a[i] + b[i], c[i]))
        
print("*** All checks complete.")


print("- Finalizing...");
ctx.destroy()
