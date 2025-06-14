#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>

#define FILTER_RADIUS 3

#define OUTPUT_TILE_DIM 32
#define INPUT_TILE_DIM ((INPUT_TILE_DIM) - 2*(INPUT_TILE_DIM))

// Constant memory to cache the filter in the GPU. There's a special command to load this into 
// GPUs constant memory called - CudaMemCpyToSymbol()
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];


// What do we need for a convolution kernel? Well it would be 2 arrays atleast
// A normal float array N and a filter F. We know that convolution is basically cracked matmul so
// Let's see how this goes
__global__ void conv2D(float* N, float* P, int width) {
    
    // This (row, col) pair is really just a threead index. 
    // For convolution the first step is to get the FILTER_RADISU * FILTER_RADIUS amount 
    // of numbers around each thread to multiply with the actual filter
    int row = blockIdx.y  * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float PValue = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
        for(int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
            int inRow = row - FILTER_RADIUS + fRow;
            int inCol = col - FILTER_RADIUS + fCol;

            if (inRow < row & inCol < col) {
                PValue += F[fRow][fCol] * N[inRow * width + fCol];
            }
        }
    }

    P[row * width + col] = PValue;
}

#define INPUT_TILE_DIM 4
#define OUT_TILE_DIM 2
#define FILTER_RADIUS 3
#define FILTER_DIM 3

__constant__ float F[FILTER_DIM][FILTER_DIM];

__global__ void shared_Conv2D(float* N, float* P, int W) {
    // Shared memory with padding for halo
    __shared__ float sh[INPUT_TILE_DIM + FILTER_RADIUS - 1][INPUT_TILE_DIM + FILTER_RADIUS - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    // Compute effective shared memory indices
    int shared_row = ty;
    int shared_col = tx;

    if (row < W && col < W) {
        sh[shared_row][shared_col] = N[row * W + col];
    } else {
        sh[shared_row][shared_col] = 0.0f;
    }

    __syncthreads();

    // Rest of convolution code goes here (not yet implemented)
}
