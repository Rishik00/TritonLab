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


    // init PValue = 0 
    float PValue = 0.0f;

    // Alright, we're around one fine grained thread, meaning one thread
    // gives me one output eleemnt. So in this case because convolution is
    // a sliding window operation, we'd want to construct a winow of size 2r + 1 * 2r + 1
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
        for(int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {

            // calculating the indexes to go backward and to the left when we are at the 
            // first eleemnt. These are also called as halo elements
            int inRow = row - FILTER_RADIUS + fRow;
            int inCol = col - FILTER_RADIUS + fCol;

            // Bound check
            if (inRow < row & inCol < col) {
                // for one p value we compute the mat mul between 2 matrices
                PValue += F[fRow][fCol] * N[inRow * width + fCol];
            }
        }
    }

    // if you cant understand this, i cant help you
    P[row * width + col] = PValue;
}

