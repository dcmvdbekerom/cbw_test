#cython: language_level=3

import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport openmp
import ctypes
from time import perf_counter
from libc.math cimport log, exp, sin, cos, tan, fabs


from crs_cpp cimport initCUDA, finalizeCUDA, setupDeviceMemory, releaseDeviceMemory, runKernel


def cy_initCUDA():
   void initCUDA()

def cy_finalizeCUDA():
    void finalizeCUDA()
    
def cy_setupDeviceMemory():
    void setupDeviceMemory(CUdeviceptr* d_a, CUdeviceptr* d_b, CUdeviceptr* d_c);
    
def cy_releaseDeviceMemory():    
    void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);

def cy_runKernel():
    void runKernel(CUdeviceptr d_a, CUdeviceptr d_b, CUdeviceptr d_c);







def cy_calc_Gamma( float p, 
                   float T, 
                   float_f[::1] params,
                   float_f[::1] EvJ_1, 
                   float_f[::1] EvJ_0,
                   float T0=1450, #TODO: shouldn't this really be T0 = 298.0K?
                   int J_min=0, 
                   int J_max=100, 
                   ):  
    cdef float_f a, alpha, beta, delta, n
    cdef Py_ssize_t Ji, Jj, J_min_k, J_max_k, J_clip, J, delta_J
    cdef float_f U1, U2, D1, D2, dE_ij, gamma_ji
    
    if float_f is float:
        dtype = np.float32
    else: # float_f is double:
        dtype = np.float64
    
    a, alpha, beta, delta, n = params
    Gamma_RPA_arr = np.zeros((5,(J_max+1)), dtype=dtype)
    cdef float_f[:,::1] Gamma_RPA = Gamma_RPA_arr
    
    for Ji in range(J_min, J_max + 1):
        U1 = ((1+((a*EvJ_0[Ji]) / (k_B*T*delta))) /
              (1+((a*EvJ_0[Ji]) / (k_B*T))))**2
    
        for Jj in range(J_min, J_max + 1):
            dE_ij = h*c*(EvJ_1[Jj] - EvJ_0[Ji])
                
            if Jj > Ji: 
                U2 = exp((-beta*dE_ij)/(k_B*T))
                D1 = (2.*Ji+1.)/(2.*Jj+1.)
                D2 = exp(dE_ij/(k_B*T))
                
                gamma_ji = U1*U2*p*alpha*((T/T0)**n)*pi*c    
                Gamma_RPA[2, Ji] += gamma_ji
                Gamma_RPA[2, Jj] += gamma_ji*D1*D2
    
    for delta_J in range(-2,3):
        if delta_J:
            J_min_k = J_min - min(delta_J, 0)
            J_max_k = J_max - max(delta_J, 0)

            for J in range(J_max + 1):
                J_clip = min(J_max, max(J_min, J))
                Gamma_RPA[delta_J+2, J_clip] = (Gamma_RPA[2,J_clip] + Gamma_RPA[2,J_clip + delta_J])*0.5
    
    return Gamma_RPA_arr


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False) 
def cy_calc_matrix_from_data(double p,
                             double T, 
                             double tau23,
                             float_f[:,::1] fdata,
                             int[:,::1] idata,
                             float_f[::1] Gamma_RPA,                      
                             double w_min, double dw, Py_ssize_t N_w,
                             double log_G_min, double dxG, Py_ssize_t N_G,
                             int chunksize=1024*128,
                             func='cpp',
                             ):
    
    cdef Py_ssize_t Nlines = <Py_ssize_t> fdata.shape[1]
    cdef Py_ssize_t l0
    cdef float_f Gi, log_Gi, l, tG, exp_Gitau
    cdef int i

    if float_f is float:
        dtype = np.float32
        dtype_c = np.complex64
    else: # float_f is double:
        dtype = np.float64
        dtype_c = np.complex128
        
    if func == 'cpp':
        f_calc = crs_cpp.cpp_calc_matrix_fp64
    else: #func == 'simd':
        f_calc = crs_cpp.simd_calc_matrix_fp64
    
    G_l_arr = np.exp(log_G_min + np.arange(N_G)*dxG).astype(dtype) # for approximate Gi
    exp_Gtau_arr = np.exp(-G_l_arr*tau23).astype(dtype)
    l0_arr = np.zeros(Gamma_RPA.size, dtype=np.int32)
    aG1_arr = np.zeros(Gamma_RPA.size, dtype=dtype)
    S_kl_arr = np.zeros((N_w, N_G), dtype=dtype_c)
    
    cdef float_f[::1] G_l = G_l_arr
    cdef float_f[::1] exp_Gtau = exp_Gtau_arr
    cdef float_f[::1] aG1_view = aG1_arr
    cdef int[::1] l0_view = l0_arr
    cdef float_f[::1] S_kl = S_kl_arr.view(dtype).reshape(N_w * N_G * 2)
       
    Wi_arr = np.zeros(Nlines, dtype=dtype)
    cdef float_f[::1] Wi_view = Wi_arr


    for i in range(Gamma_RPA.size):
        Gi = Gamma_RPA[i]
        log_Gi = log(Gi)
        l = (log_Gi - log_G_min) / dxG
        l0 = <Py_ssize_t>l
        l0_view[i] = l0
        
        tG = l - l0    
        exp_Gitau = exp(-Gi*tau23)
        aG1_view[i] = (exp_Gitau - exp_Gtau[l0]) / (exp_Gtau[l0 + 1] - exp_Gtau[l0])
        
    #crs_cpp.cpp_calc_matrix_fp64( 
    f_calc(
           <double> p,
           <double> T, 
           <double> tau23,
           <double*> &fdata[0,0],
           <double*> &fdata[1,0],
           <double*> &fdata[2,0],
           <int*> &idata[0,0],
           <int*> &l0_view[0],
           <double*> &aG1_view[0],                      
           <double> w_min, 
           <double> dw, 
           <int> N_w,
           <int> N_G,
           <int> chunksize,
           <int> Nlines,
           <double*> &Wi_view[0],       
           <double*> &S_kl[0],
           )
   
    return Wi_arr, S_kl_arr 
    
from scipy.fft import ifft, fft    


def cy_test_complex(double[::1] arr):
    print(arr.shape)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)     
def cy_test_mult( double tau23,
                  double[::1] w_arr,
                  np.ndarray[np.complex128_t, ndim=2] S_kl,
                  np.ndarray[np.complex128_t, ndim=2] S_kl_a,
                  func='cpp'):
    
    N_t = S_kl_a.shape[1]
    N_G = S_kl_a.shape[0]    
    cdef double[:,::1] S_kl_view = S_kl.view(np.float64)
    cdef double[:,::1] S_kl_a_view = S_kl_a.view(np.float64)
    
    if func == 'cpp':
        f_calc = crs_cpp.cpp_mult1_fp64
    else: #func == 'simd':
        f_calc = crs_cpp.simd_mult1_fp64
    
    f_calc( <int> N_t,
            <int> N_G,
            <double> tau23,
            <double*> &w_arr[0],
            <double*> &S_kl_view[0,0],
            <double*> &S_kl_a_view[0,0])
    
    return S_kl_a


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)     
def cy_calc_transform(double tau23,
                      double[::1] w_arr,
                      double[::1] t_arr,
                      double log_G_min,
                      double dxG,
                      np.ndarray[np.complex128_t, ndim=2] S_kl,
                      np.ndarray[np.complex128_t, ndim=2] S_kl_a,
                      double[::1] E_probe,
                      times):
    cdef int N_G, N_t, k, l
    cdef double theta, sr, si, Sr, Si, G_l, factor, chi_r, chi_i
    
    N_t = S_kl.shape[0]
    N_G = S_kl.shape[1]
    
    E_CARS = np.zeros(N_t, dtype=np.complex128)
    I_CARS = np.zeros(N_t, dtype=np.float64)
    
    cdef double[:,::1] S_kl_view = S_kl.view(np.float64)
    cdef double[:,::1] S_kl_a_view = S_kl_a.view(np.float64)
    cdef double[::1]   E_CARS_view = E_CARS.view(np.float64)
    cdef double[::1]   I_CARS_view = I_CARS

    toc = tic()
    for k in range(N_t):
        theta = tau23*w_arr[k]
        sr = cos(theta)
        si = sin(theta)
        
        for l in range(N_G):
            Sr = S_kl_view[k,2*l  ]
            Si = S_kl_view[k,2*l+1]
            
            S_kl_a_view[l,2*k  ] = Sr*sr - Si*si
            S_kl_a_view[l,2*k+1] = Sr*si + Si*sr
            
    # cpp_mult1_fp64( <int> N_t,
                    # <int> N_G,
                    # <double> tau23,
                    # <double*> &w_arr[0],
                    # <double*> &S_kl_view[0,0],
                    # <double*> &S_kl_a_view[0,0])
                    
    times['Mult1'] = toc.val()*1e3
    
    toc = tic()
    ifft(S_kl_a, axis=1, overwrite_x=True, workers=8)
    times['IFFT'] = toc.val()*1e3

    toc = tic()
    # for l in range(N_G):
        # G_l = exp(log_G_min + l*dxG)
        # for k in range(N_t):
            # factor = exp(-G_l * (t_arr[k] + tau23)) * N_t
            
            # chi_r = S_kl_a_view[l, 2*k  ] * factor
            # chi_i = S_kl_a_view[l, 2*k+1] * factor
            
            # E_CARS_view[2*k  ] += chi_r * E_probe[k]
            # E_CARS_view[2*k+1] += chi_i * E_probe[k]
    
    simd_mult2_fp64(<int> N_G,
                    <int> N_t,
                    <double> tau23,
                    <double> log_G_min,
                    <double> dxG,                
                    <double*> &E_probe[0],
                    <double*> &t_arr[0],
                    <double*> &S_kl_a_view[0,0],
                    <double*> &E_CARS_view[0])
    
    times['Mult2'] = toc.val()*1e3

    toc=tic()
    fft(E_CARS, overwrite_x=True, workers=8)
    times['FFT'] = toc.val()*1e3

    toc = tic() 
    for k in range(N_t):
        I_CARS_view[k] = E_CARS_view[2*k]**2 + E_CARS_view[2*k+1]**2 
    times['Norm'] = toc.val()*1e3

    return I_CARS
                      