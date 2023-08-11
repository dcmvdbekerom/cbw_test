
extern "C" {

    struct transform {
        int offset;
        int scale;
    };

    __device__ __constant__ size_t N;
    __device__ __constant__ transform params;
  
    __global__ void matSum(int *a, int *b, int *c){
        int tid = blockIdx.x;
        if (tid < N)
            c[tid] = (a[tid] + b[tid]) * params.scale + params.offset;
        }
}