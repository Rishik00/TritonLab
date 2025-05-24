#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>
#include <math.h>

#define THREADS_PER_BLOCK 8
#define BLOCKS_PER_GRID 1
#define SIZE 8
#define REDUNDANT 0.0f

__global__ void shared_sigmoid_kernel(float* M) {

    __shared__ float sdata[THREADS_PER_BLOCK];
    int tidx = threadIdx.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (global_idx < SIZE) {
        sdata[tidx] = M[global_idx];
    } else {
        sdata[tidx] = REDUNDANT;
    }

    if (tidx < SIZE) {
        sdata[tidx] = 1 / (1 + (expf(-sdata[tidx])) );
    }
    __syncthreads();

    if (tidx < SIZE) {
        M[global_idx] = sdata[tidx];
    }

}

void apply_shared_sigmoid () {
    float* Mh = new float[SIZE];
    float* Md;

    for (int i=0; i < SIZE; i++) {
        Mh[i] = 0.87*i;
    }

    std::cout << "Input: " << std::endl;
    for(int j=0;j<SIZE;j++) {
        std::cout << Mh[j] << ' ';
    }
    std::cout << std::endl;

    cudaMalloc(&Md, SIZE * sizeof(float));
    cudaMemcpy(Md, Mh, SIZE * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 blocksInGrid(BLOCKS_PER_GRID);

    shared_sigmoid_kernel<<<blocksInGrid, threadsPerBlock>>>(Md);
    cudaMemcpy(Mh, Md, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output: " << std::endl;
    for(int j=0;j<SIZE;j++) {
        std::cout << Mh[j] << ' ';
    }
    std::cout << std::endl;
    
    cudaFree(Md);
    delete[] Mh;
}

int main() {
    apply_shared_sigmoid();
    return 0;
}