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
#define VECTOR_SIZE 10

// each block will be os size (VECTOR_SIZE, 1)

// Take in a matrix, and reduce it to a vector of size NUUM_OLUMNS
__global__ void Max2DRedOnAxis(
    float *M, 
    float *redVector, 
    int    axis
) {

    int bid, tid, block, idx;

    if (axis == 0) {
        idx = threadIdx.x; + blockIdx.x; * blockDim.x;

        float pSum = 0.0f;
        for (int m = 0; m <= VECTOR_SIZE; m++) {
            int vecIdx = idx + m * VECTOR_SIZE;

            psum = psum + M[vecIdx];
        }


    } else {
        idx = threadIdx.y + blockIdx.y; * blockDim.y;
        
        float pSum = 0.0f;
        for (int m = 0; m <= VECTOR_SIZE; m++) {
            int vecIdx = idx + m;

            psum = psum + M[vecIdx];
        }

    }

}