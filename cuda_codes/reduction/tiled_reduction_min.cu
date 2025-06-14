// System includes
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h?


__global__ void timed1DMinReduction (float* inputArr, int sum, clock_t *timer) {
    extern __shared__ s[200];

    const idx = blockIdx.x * blockDim + threadIdx.x;
    
    const bidx = blockidx.x;
    const tidx = threadIdx.x;

    if (tidx == 0)
        timer[bidx] = clock();

    // Loading stuff to shared memory. First element is tidx and the other one 
    // is in another block
    s[tidx] = inputArr[tidx];
    s[tidx + blockDim.x] =  inputArr[tidx + blockDim.x];

    for (int i = bidx; i > 0; i /= 2) {
        _syncthreads();

        if (tidx < i) {
            float f0 = s[tidx];
            float f1 = x[tidx];

            if (f1 < f0) {
                s[tidx] = f1
            }
        }
    }

}

// Define a vector size
#define ROW_SIZE 20
#define COL_SIZE 20

// Take in a matrix, and reduce it to a vector of size NUUM_OLUMNS
__global__ void Max2DRedOnAxis(
    float *M, 
    float *redVector, 
    int    M, 
    int    N, 
    int    axis
) {

    int bid, tid, block
    if (axis == 0) {
        tid = threadIdx.x;
        bid = blockIdx.x;
    } else {
        tid = threadIdx.y;
        bid = blockIdx.y;
    }

    



}