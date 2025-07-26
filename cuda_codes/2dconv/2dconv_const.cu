#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>

#define INPUT_TILE_DIM 4
#define OUT_TILE_DIM 2
#define FILTER_RADIUS 3
#define FILTER_DIM 3

// this is defining a constant for the GPU
// We're squeezing out everything from this GPU now haha
// Basically CUDA has a constant memory pool that is a lot faster than global memory and isnt super volatile like
// shared memory
__constant__ float F[FILTER_DIM][FILTER_DIM];

// But dw, we will be using shared memory to load 
// Out matrix's elements and compute faster

__global__ void shared2Dconv(float* N, float* P, int W, int H) {

    // The indexes of doom. 
    int row = blockIdx.y  * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Lets define some shared memory tiles 
    // to load our input into
    // Recall that shared memory is static and we dont need 
    // all the jacking off related to linear indexing now :)
    __shared__ float Nds[INPUT_TILE_DIM][INPUT_TILE_DIM]

    // Bound check 
    if (row >= 0 && row <= W && col >= 0 && col <= W) {
        Nds[threadIdx.x][threadIdx.y] = N[row * W  + col];
    } else {
        Nds[threadIdx.x][threadIdx.y] = 0.0f;
    }

    // because we had branch divergence there, we want to sync 
    // all the threads after any kind of branching.
    __syncthreads();

    // Defining the tile starting to be to the top right of the first element, 
    // ig we are trying to construct that matrix now? 
    int tileCol = threadIdx.y - FILTER_DIM;
    int tielROw = threadIdx.x - FILTER_DIM;

    // But how is tileRow + fRow == threadIdx.x? 
    if (row >= 0 && row <= W && col >= 0 && col <= W) {
        Pval = 0.0f;
        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
            for(int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                
                
                Pval += F[fRow][fCol] * Nds[tileRow + fRow][tileCol + fCol];
            }
        }
        P[row * width + col] = PVal;   
    }



}

void kernelSetup(){ 

}

void main() {

}