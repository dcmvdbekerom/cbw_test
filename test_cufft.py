import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfftfreq, rfft, irfft, fftshift, ifftshift

L = lambda t, w: 2/(w*np.pi) * 1/(1 + 4*(t/w)**2)
L_FT = lambda f, w: np.exp(-np.pi*np.abs(f)*w)


t_max = 100.0
dt = 0.001
t_arr = np.arange(-t_max, t_max, dt)
Nt = len(t_arr)

w0 = 1.0
I_arr = L(t_arr, w0)

f_arr = rfftfreq(Nt, dt)
I_FT_arr = rfft(fftshift(I_arr*dt)).real

fig, ax = plt.subplots(1,2)

ax[0].axhline(0,c='k',alpha=0.5)
ax[0].plot(t_arr, I_arr)
ax[0].set_xlim(-5,5)

ax[1].axhline(0,c='k',alpha=0.5)
ax[1].plot(f_arr, I_FT_arr)
ax[1].plot(f_arr, L_FT(f_arr, w0), 'k--')
ax[1].set_xlim(0,2)

plt.show()
