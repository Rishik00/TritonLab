#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>
#include <math.h>

#define THREADS_PER_BLOCK 8
#define BLOCKS_PER_GRID 1
#define SIZE 8

// Stage - 1: exponential function
__global__ void vector_exponential_kernel(float* M, int N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx < N) {
        M[tidx] = expf(M[tidx]);
    }
}

// Stage - 2: Shared memory used sum 
__global__ void vector_exponential_sum_kernel(float* M, float* sum_d, int N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float sdata[THREADS_PER_BLOCK];

    if (global_idx < N) {
        sdata[tidx] = M[global_idx];
    } else {
        sdata[tidx] = 0.0f;
    }

    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
        if (tidx < stride) {
            sdata[tidx] += sdata[tidx + stride];
        }
        __syncthreads();
    } 
    if (tidx == 0) {
        *sum_d = sdata[0];
    }

}

// Stage - 3: vector normalization using stage 1 and 2 outputs
__global__ void vector_normalization_kernel(float* M, float* P, float* sum, int N) {
    int tidx = threadIdx.x;

    if (tidx < N) {
        P[tidx] = M[tidx] / *sum;
    }
}


void apply_softmax_kernel(int N) {
    float* Mh = new float[N];
    float* Ph = new float[N];
    float* sum_h = new float;

    float* Md, *sum_d, *Pd;

    for (int i=0; i < N; i++) {
        Mh[i] = 0.87*i;
    }

    std::cout << "Input: " << std::endl;
    for(int j=0;j<N;j++) {
        std::cout << Mh[j] << ' ';
    }
    std::cout << std::endl;
    
    cudaMalloc(&Md, N * sizeof(float));
    cudaMalloc(&Pd, N * sizeof(float));
    cudaMalloc(&sum_d, sizeof(float));

    cudaMemcpy(Md, Mh, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);
    dim3 blocksInGrid(BLOCKS_PER_GRID);
    
    vector_exponential_kernel<<<blocksInGrid, threadsPerBlock>>>(Md, N);

    vector_exponential_sum_kernel<<<blocksInGrid, threadsPerBlock>>>(Md, sum_d, N);
    cudaMemcpy(sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost);

    vector_normalization_kernel<<<blocksInGrid, threadsPerBlock>>>(Md, Pd, sum_d, N);

    cudaMemcpy(Ph, Pd, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Pd);
    cudaFree(sum_d);

    std::cout << "Done: " << *sum_h << std::endl;
    std::cout << "Output: " << std::endl;
    for(int j=0;j<N;j++) {
        std::cout << Ph[j] << ' ';
    }
    std::cout << std::endl;

    delete[] Ph;
    delete[] Mh;
    delete sum_h;
}

int main() {

    apply_softmax_kernel(SIZE);
    return 0;

}
