import numpy as np
from cuda_driver import CuContext, CuArray, CuModule
from ctypes import c_int, c_longlong, Structure
from time import sleep

N = 100
dtype = np.int32

print("- Initializing...");
for i, dev in enumerate(CuContext.getDeviceList()):
    print('Device {:d}: {:s}'.format(i,dev))
ctx = CuContext()
ctx.printDeviceCapabilities()


class transform(Structure):
    _fields_ = [
        ("offset", c_int),
        ("scale", c_int),
        ]

params = transform()
params.scale = 2
params.offset = 3

a_h = N - np.arange(N, dtype=dtype)
b_h = np.arange(N, dtype=dtype)**2

a_d = CuArray.fromArray(a_h)
b_d = CuArray.fromArray(b_h)
c_d = CuArray(N, dtype=dtype, init='empty')


mod = CuModule(ctx, "cu/matSumKernel.ptx")
mod.matSum.set_grid(blocks=(N,1,1))
mod.setConstant('N', c_longlong(N))
mod.setConstant('params', params)
#ctx.synchronize()

print("# Running the kernel...")
mod.matSum(a_d, b_d, c_d)
c_h = c_d.getArray()
print("# Kernel complete.")

for i in range(N):
    if (c_h[i] != (a_h[i] + b_h[i])*params.scale + params.offset):
        print("* Error at array position {:d}: Expected {:d}, Got {:d}".format(
            i, (a_h[i] + b_h[i])*params.scale + params.offset, c_h[i]))
        
print("*** All checks complete.")


print("- Finalizing...");
ctx.destroy()
