# -*- coding: utf-8 -*-
"""
Created on Tue May  9 16:54:53 2023

- v1.0: * Load database from numpy files; load chi_ref.npy and I_CARS_ref.npy to benchmark results

- v1.1: * Rewrite Gamma's in more general index

- v1.2: * Combine all branches into single database & calculations; 
        * Moved some functions to crs_lib.py
        * Separate omega/freq/nu and generate by fftfreq

- v1.3: * Implement DIT with 4pt weight for position and simple weight for Gamma
        * Implement DIT with position and dampening weights optimized at tau23

- v1.4: * Removed weird complex gaussian term without problems
        * Introduced discrepancy of 0.012cm-1 in the spectral axis; appears to be issue with benchmark data
        * Center transform around tau23 and reduce temporal extent to minimum allowable
        
- v1.5: * Store normalized sigma as npy file
        * Further refine Gamma calculation
        * Include one less line in cutoff, making indexing siginifcantly simpler; no longer sorts arrays.
        * Increase range to 2001.0 cm-1  to solve issue with full database

- v1.6: * Make cython functions for stick spectrum and matrix calculation
        * Cambine functions into a single one

- v1.7: * Sort by line postition to reduce likelihood of overlapping threads
        * Remove live line filtering
        * Move all temperature dependent parts into single function
        * Add slider widget for interactive plotting

- v1.7b * Found precise offset; normalization now also included

- v1.8: * Indexing through slicing


@author: bekeromdcmvd
"""

#%% Imports:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.constants import pi, c
c *= 100 #cm.s-1
from scipy.fft import next_fast_len, fft, ifft, fftfreq
import sys
sys.path.append('./cython')
from cy_crs import cy_calc_matrix_from_data, cy_calc_Gamma, tic, cy_calc_transform, cy_test_mult
from scipy.interpolate import CubicSpline
import pyfftw


def load_aligned(fname, default=0.0, n=256, dtype=np.float64):
    byte_align = n // 8
    elem_align = n // np.dtype(dtype).itemsize
    
    load_arr = np.load(fname, mmap_mode='r')
    ret_arr = np.zeros(load_arr.size + elem_align, dtype=dtype)
    offset = (ret_arr.ctypes.data % byte_align) // np.dtype(dtype).itemsize
    ret_arr[offset : offset + load_arr.size] = load_arr[:]
    
    if default == 'first':
        default = ret_arr[offset]
    if default == 'last':
        default = ret_arr[offset + load_arr.size - 1]
        
    ret_arr[offset + load_arr.size:] = default
    ret_size = int(np.ceil(load_arr.size / elem_align) * elem_align)
    
    return ret_arr[offset, offset + ret_size]


def next_fast_aligned_len(n_in, elem_align=4):
    n = next_fast_len(n_in)
    while(n % elem_align):
        n = next_fast_len(n + 1)
    return n



#%% Import data
print('Loading database... ', end='')
toc = tic()

nu_data = np.load('data/nu_data.npy')
J_data = np.load('data/J_data.npy')
E0_data = np.load('data/E0_data.npy')
gR_data = np.load('data/gR_data.npy')
branch_data = np.load('data/branch_data.npy')
sigma_data = np.load('data/sigma_data.npy')


#crop arrays to sets of 4
vec_size = 4
N_simd = ((1 << (4*8)) - vec_size) & len(nu_data)

print('Done! [{:.1f}M lines loaded]'.format(len(nu_data)*1e-6))


#%% Initialization parameters:

p = 1.0 #bar  
T = 296.0 #K   

##### MEG fitting parameters (placeholders) #####
a = 2;                                                                         # species-specific constant set to 2 
alpha = 0.0445;
beta = 1.52;
delta = 1;                                                                     # room-temperature value
n = 0;     

# CH_4 v2 ro-vibrational constants from Olafson 1962 [cm-1]
B0 = 5.2412;                                                                   # rotational constant, v=0 [cm-1]
D0 = 1.1e-4;                                                                   # centrifugal constant, v=1 rot. sublevel alpha
B1_b = 5.379;                                                                  # rot.constant, v=1 rot. sublevel beta (O & S branches)
D1_b = 1.7e-4;  

J_Q = J_data[:,branch_data==b'Q']
J_min = np.min(J_Q[0]) #TODO: Shouldn't J start at 0?
J_max = np.max(J_Q[0])

calc_EvJ_1 = lambda J: ( (B1_b*J*(J+1)) - (D1_b*J**2 * (J+1)**2) )             # upper ro-vibrational state energy
calc_EvJ_0 = lambda J: ( (B0  *J*(J+1)) - (D0  *J**2 * (J+1)**2) );            # lower ro-vibrational state energy

dt_pr = 5e-12; #FWHM of the probe
tau = (np.arange(25, 250+1, 1))*1e-12;   
itau = 142
tau23 = tau[itau]

N_FWHM = 100
t_max = N_FWHM * dt_pr
v_min = 1100.0 #cm-1
v_max = 2000.0 #cm-1

dxG = 0.2 # Resolution of Dampening-grid
dtype = np.float64


#%% Init data:
dt = 1/(c*(v_max - v_min)) #s
t_max = N_FWHM*dt_pr

N_t = next_fast_aligned_len(int(2*t_max/dt))
t_arr = fftfreq(N_t, d=1/(2*t_max)) #s

N_v = N_t
N_w = N_v
v_arr = np.linspace(v_min, v_max, N_v, endpoint=False)
dv = (v_arr[-1] - v_arr[0]) / (N_v - 1)
dw = 2*pi*c*dv
w_min = 2*pi*c*v_min
w_arr = 2*pi*c*v_arr

delta_J = branch_data.view(np.int8) - ord('Q')
fdata = np.zeros((3, N_simd), dtype=dtype)
fdata[0] = nu_data[:N_simd]
fdata[1] = (sigma_data * np.min(gR_data, 0))[:N_simd]
fdata[2] = E0_data[:N_simd] 

J_min_k = J_min - np.clip(delta_J, None, 0)
J_max_k = J_max - np.clip(delta_J, 0, None)
J_clip = np.clip(J_data[1,:], J_min_k, J_max_k)

idata = np.zeros((2, N_simd), dtype=np.int32)
idata[0] = ((delta_J+2)*(J_max+1) + J_clip)[:N_simd]
idata[1] = 0#delta_J[:N_simd]

J_arr = np.arange(J_max + 1)
EvJ_1 = calc_EvJ_1(J_arr)
EvJ_0 = calc_EvJ_0(J_arr)

params = np.array([a, alpha, beta, delta, n], dtype=np.float64)
Gamma_RPA = cy_calc_Gamma(p, T, params, EvJ_1, EvJ_0, 
                      J_min=J_min, J_max=J_max)

log_G_min = np.min(np.log(Gamma_RPA[2,J_min:J_max+1]))
log_G_max = np.max(np.log(Gamma_RPA[2,J_min:J_max+1]))  

N_G = int(np.ceil((log_G_max - log_G_min)/dxG) + 1)

Wi_cpp, S_kl_cpp = cy_calc_matrix_from_data(p, T, tau23, 
                                fdata[:,:N_simd], #TODO: make this prettier?
                                idata[:,:N_simd],  
                                Gamma_RPA.reshape(5*(J_max+1)),
                                w_min, dw, N_w,
                                log_G_min, dxG, N_G,
                                chunksize=1024*128,
                                func='cpp',
                                )

Wi_simd, S_kl_simd = cy_calc_matrix_from_data(p, T, tau23, 
                                fdata, #TODO: make this prettier?
                                idata,  
                                Gamma_RPA.reshape(5*(J_max+1)),
                                w_min, dw, N_w,
                                log_G_min, dxG, N_G,
                                chunksize=1024*128,
                                func='simd',
                                )


Wi_cpp = Wi_cpp.astype(np.uint64)
Wi_simd = Wi_simd.astype(np.uint64)

S_kl_a = pyfftw.empty_aligned((N_G,N_w), dtype=np.complex128, n=256)
S_kl_FT_a = pyfftw.empty_aligned((N_G,N_w), dtype=np.complex128, n=256)

E_CARS_FT_a = pyfftw.empty_aligned(N_w, dtype=np.complex128, n=256)
E_CARS_a = pyfftw.empty_aligned(N_w, dtype=np.complex128, n=256)


pyfftw.config.NUM_THREADS = 4
pyfftw.config.PLANNER_EFFORT = 'FFTW_EXHAUSTIVE'

# toc = tic()    
# ifft_obj = pyfftw.FFTW(S_kl_a, S_kl_FT_a, direction='FFTW_BACKWARD', axes=(1,))
# fft_obj = pyfftw.FFTW(E_CARS_FT_a, E_CARS_a, direction='FFTW_FORWARD')
# print('Planning FFT:', toc.val()*1e3)


times = {}
toc = tic()
shift_arr = np.exp(2j*pi*c*tau23*v_arr)    
for l in range(N_G):
    S_kl_a[l,:] = S_kl_simd[:,l]*shift_arr
times['Mult1'] = toc.val()*1e3

toc = tic()
ifft(S_kl_a, axis=1, overwrite_x=True, workers=8)
times['IFFT'] = toc.val()*1e3

E_probe = np.exp(-(2*np.log(2)*(t_arr)/dt_pr)**2)*dt      

# E_cpp = cy_test_mult(tau23,
#                     w_arr,
#                     t_arr,
#                     log_G_min,
#                     dxG,
#                     S_kl_a,
#                     E_probe,
#                     times,
#                     func='cpp')

# E_simd = cy_test_mult(tau23,
#                     w_arr,
#                     t_arr,
#                     log_G_min,
#                     dxG,
#                     S_kl_a,
#                     E_probe,
#                     times,
#                     func='simd')

# print(E_cpp)
# print(E_simd)

# print(np.allclose(E_cpp, E_simd))#, atol=1e-10, rtol=0.0))


# sys.exit()











#base_addr0 = [ 54002.  54010.  54006. ... 485986. 485992. 485996.]


# for i in range(0,4000,4*100):
#     print(Wi_cpp[i] - S_kl_cpp.ctypes.data, Wi_simd[i] - S_kl_simd.ctypes.data)

# print(S_kl_cpp)
# print(S_kl_simd)

i_max = 2678025
# print(np.allclose(Wi_cpp[:i_max], Wi_simd[:i_max]))#, atol=1e-10, rtol=0.0))
# print(np.allclose(S_kl_cpp[:,:i_max], S_kl_simd[:,:i_max]))#, atol=1e-10, rtol=0.0))

#print(np.arange(Wi_cpp.size)[Wi_cpp!=Wi_simd]) # only for ints



#%% Spectral calculation function definition:
    
def calc_spectrum(p, T, tau23, dxG, params):
    # global log_G_min, log_G_max, Gamma_JJ, log_Gamma_RPA, log_G_min2, log_G_max2
    
    times = {}
    toc = tic()    
    Gamma_RPA = cy_calc_Gamma(p, T, params, EvJ_1, EvJ_0, 
                          J_min=J_min, J_max=J_max)
    
    # Only look for extremes along the Q-branch.
    # This will yield correct results because the others are averages (so less extreme)
    log_G_min = np.min(np.log(Gamma_RPA[2, J_min:J_max+1]))
    log_G_max = np.max(np.log(Gamma_RPA[2, J_min:J_max+1]))  
    
    N_G = int(np.ceil((log_G_max - log_G_min)/dxG) + 1)

    times['Gamma'] = toc.val()*1e3
    
    toc = tic()     
    # S_kl = cy_calc_matrix_from_data(p, T, tau23, 
    #                                 fdata, #TODO: make this prettier?
    #                                 idata,  
    #                                 Gamma_JJ*0.5, #pre-multiply by 0.5 for RPA
    #                                 w_min, dw, N_w,
    #                                 log_G_min, dxG, N_G,
    #                                 chunksize=1024*128,
    #                                 )
    
    Wi_simd, S_kl = cy_calc_matrix_from_data(p, T, tau23, 
                                fdata, #TODO: make this prettier?
                                idata,  
                                Gamma_RPA.reshape(5*(J_max+1)),
                                w_min, dw, N_w,
                                log_G_min, dxG, N_G,
                                chunksize=1024*16,
                                func='simd',
                                )
    times['Dist'] = toc.val()*1e3
    

    I_CARS = cy_calc_transform(tau23,
                      w_arr,
                      t_arr,
                      log_G_min,
                      dxG,
                      S_kl,
                      S_kl_a,
                      E_probe,
                      times)
    
    
    
    # toc = tic()
    # shift_arr = np.exp(2j*pi*c*tau23*v_arr)    
    # for l in range(N_G):
    #     S_kl_a[l,:] = S_kl[:,l]*shift_arr
    # times['Mult1'] = toc.val()*1e3
    
    # toc = tic()
    # ifft(S_kl_a, axis=1, overwrite_x=True, workers=8)
    # times['IFFT'] = toc.val()*1e3
    
    # toc = tic() 
    # chi = np.zeros(N_t, dtype=np.complex128)
    # for l in range(N_G):
    #     G_l = np.exp(log_G_min + l*dxG)
    #     chi += S_kl_a[l,:] * np.exp(-G_l * (t_arr + tau23))*N_t
    # E_CARS = chi*E_probe   # CRS envelope: chi x probe envelope
    
    # times['Mult2'] = toc.val()*1e3
    
    # toc = tic()
    # # fft_obj()
    # fft(E_CARS, overwrite_x=True, workers=8)
    # times['FFT'] = toc.val()*1e3
    
    # toc=tic()
    # CARS_signal = np.abs(E_CARS);                                  
    # I_CARS = CARS_signal**2
    # times['Norm'] = toc.val()*1e3
    

    I_CARS /= np.max(I_CARS)            # <--
    
    return I_CARS, times


#%% Reference spectrum
print('Loading reference spectrum... ',end='')

# This is purely for legacy purposes, because we need the old spectral axis:
_timeframe = np.arange(0,1e3+8e-3,8e-3)*1e-12;       #s                        # timegrid
_N_t = len(_timeframe);
_dt = (_timeframe[-1] - _timeframe[0])/(_N_t-1) 
_freq = np.fft.fftshift(np.fft.fftfreq(_N_t, d=_dt)) #Hz
_omega = 2*pi*_freq #s-1
_nu = _freq[_N_t//2:]/c #cm-1
_nu_corr = 0.01115#cm-1
_nu += _nu_corr
_dnu = _nu[1] - _nu[0]

idx_ref = (_nu >= v_min) & (_nu < v_max)
_nu = _nu[idx_ref]

I_CARS_ref = np.load('data/I_CARS_ref_cropped.npy')

Imax = np.max(I_CARS_ref[itau])
print('Done!')


#%% Plotting:
    
#independent of # of time points
fig, ax = plt.subplots(2, sharex=True)

I_CARS, times = calc_spectrum(p, T, tau23, dxG, params)
p1, = ax[0].plot(v_arr,I_CARS, label='New implementation')    
p2, = ax[0].plot(_nu,I_CARS_ref[itau,:]/Imax, #             # <--
                  'k--', label='Benchmark')
ax[0].legend(loc=1)
# ax[0].set_yscale('log')

rtol = 1e-3
ax[1].axhline(100*rtol, c='k', ls='--')

err = np.abs(np.interp(v_arr, _nu, I_CARS_ref[itau,:])/Imax - I_CARS)#  #<--
p3,= ax[1].plot(v_arr, 100*err)
ax[1].set_ylabel('Error w.r.t. benchmark (% of max.)')

ax[1].set_xlim(v_min, v_max)
ax[1].set_ylim(-0.01, 0.3)


s1_ax = plt.axes([0.9, 0.15,  0.03, 0.7])
s1  = Slider(s1_ax, 'i', 0, len(tau)-1, valinit=itau, 
            valstep=1, orientation='vertical')

s2_ax = plt.axes([0.95, 0.15,  0.03, 0.7])
s2  = Slider(s2_ax, 'T', T, 2000.0, valinit=T, orientation='vertical')

plt.subplots_adjust(right=0.85)


def update(val):
    itau = s1.val
    T = s2.val
    tau23 = tau[itau]
    I_CARS, times = calc_spectrum(p, T, tau23, dxG, params)

    Imax = np.max(I_CARS_ref[itau])
    spl = CubicSpline(_nu, I_CARS_ref[itau])
    err = np.abs(spl(v_arr)/Imax - I_CARS)      # <--
    
    p1.set_ydata(I_CARS)
    p2.set_ydata(I_CARS_ref[itau,:]/Imax)       # <--
    p3.set_ydata(100*err)
    # ax[1].set_ylim(np.min(err)*100,np.max(err)*100)
        
    #td = times['Conv'] + times['FT']
    td = times['Mult1']
    tr = np.sum([*times.values()])
    ax[0].set_title('τ = {:4.0f} ps,  t = {:5.1f} / {:5.1f} ms'.format(tau23*1e12, td, tr))
    # ax[0].set_ylim(-Imax*0.1,Imax*1.1)
    ax[0].legend(loc=1, labels=times.keys())
    fig.canvas.draw_idle()

s1.on_changed(update)
s2.on_changed(update)
plt.show()






