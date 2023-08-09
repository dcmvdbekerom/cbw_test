
#%% Imports:

sys.path.append('./cython')
from cy_cbw import cy_initCUDA, cy_finalizeCUDA, cy_setupDeviceMemory, cy_releaseDeviceMemory, cy_runKernel
import numpy as np


##int a[N], b[N], c[N];
##CUdeviceptr d_a, d_b, d_c;

#initialize host arrays
N = 100;
i_arr = np.arange(N)
a_arr = N - i_arr
b_arr = i_arr**2

#initialize
print("- Initializing...\n");
initCUDA();

#allocate memory
cy_setupDeviceMemory(&d_a, &d_b, &d_c)

#copy arrays to device
#checkCudaErrors(cuMemcpyHtoD(d_a, a, sizeof(int) * N));
#checkCudaErrors(cuMemcpyHtoD(d_b, b, sizeof(int) * N));

#run
print("# Running the kernel...\n")
cy_runKernel(d_a, d_b, d_c)
print("# Kernel complete.\n")

#copy results to host and report
##checkCudaErrors(cuMemcpyDtoH(c, d_c, sizeof(int) * N));
##for (int i = 0; i < N; ++i) {
##    if (c[i] != a[i] + b[i])
##        print("* Error at array position %d: Expected %d, Got %d\n",
##            i, a[i] + b[i], c[i])
##}
print("*** All checks complete.\n")


#finish
print("- Finalizing...\n")
cy_releaseDeviceMemory(d_a, d_b, d_c)
cy_finalizeCUDA()
