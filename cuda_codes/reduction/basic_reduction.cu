#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>

__device__ float max_k (float a, float b) {
    if (a > b) return a;
    else return b; 
}

__device__ float min_k (float a, float b) {
    if (a < b) return a;
    else return b;
}

__global__ void simpleMaxRed(float* input, int length, float* max) {
    unsigned int idx = 2 * threadIdx.x;

    float maxValue = 0.0f;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {

            maxValue = max_k(input[idx], input[idx + stride]);
            __syncthreads();

        }
        __syncthreads();
    }

    if (idx == 0) {
        *max = input[0];
    }

}

__global__ void simpleMinRed(float* input, int length, float* min) {
    unsigned int idx = 2 * threadIdx.x;
    float minValue = 0.0f;

    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride) {
            minValue = min_k(input[idx], input[idx + stride]);
            __syncthreads();
        }
        __syncthreads();
    }
    
    if (idx == 0) {
        *min = input[0];
    }

}

void RedInit() {
    const float *arr = new float[]{3.4,5.6,7.8,1.1, 1.2, 9.22,9.23};
    const int length = 7;
    float *max;

    float *arr_d, *max_d;

    cudaMalloc((void **)&arr_d, length * sizeof(float));
    cudaMalloc((void **)&max_d, sizeof(float));
	
    cudaMemcpy(arr_d, arr, length * sizeof(float), cudaMemcpyHostToDevice);
	
    unsigned int numBlocks = 2;    
    const unsigned int numThreads = (length / numBlocks) + 1;

	simpleMinRed<<<numBlocks, numThreads>>>(arr_d, length, max_d);

	// Once the exec is done, we move it back from device to host
	cudaMemcpy(max, max_d, sizeof(float), cudaMemcpyDeviceToHost);
	
    std::cout << "Output: " << *max << std::endl;

    cudaFree(arr_d);
    cudaFree(max_d);

}

int main() {
    RedInit();
    return 0; 
}