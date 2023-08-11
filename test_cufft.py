import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift
from cuda_driver import cuContext, cuArray, cuModule, cuFFT

L = lambda t, w: 2/(w*np.pi) * 1/(1 + 4*(t/w)**2)
L_FT = lambda f, w: np.exp(-np.pi*np.abs(f)*w)

t_max = 100.0
dt = 0.001
t_arr = np.arange(-t_max, t_max, dt)
Nt = len(t_arr)

w0 = 1.0
I_arr = L(t_arr, w0)

f_arr = rfftfreq(Nt, dt)
Nf = len(f_arr)
I_FT_arr = rfft(fftshift(I_arr)*dt).real



#CUDA application:
ctx = cuContext()
cu_I_arr = cuArray(fftshift(I_arr.astype(np.float32)*dt))
cu_I_FT_arr = cuArray(np.zeros(Nf, dtype=np.complex64))

cufft_kernel = cuFFT(cu_I_arr, cu_I_FT_arr)
cufft_kernel.execute()


fig, ax = plt.subplots(1,2)

ax[0].axhline(0,c='k',alpha=0.5)
ax[0].plot(t_arr, I_arr)
ax[0].set_xlim(-5,5)

ax[1].axhline(0,c='k',alpha=0.5)
ax[1].plot(f_arr, I_FT_arr, label='Numpy.fft')
#ax[1].plot(f_arr, L_FT(f_arr, w0), 'k--')
ax[1].plot(f_arr, cu_I_FT_arr.d2h().real, 'k--', label='cuFFT')
ax[1].set_xlim(0,2)
ax[1].legend()
plt.show()
