#include <iostream>
#include <cuda_runtime.h>

constexpr int THREAD_DIM = 1024;
constexpr int BLOCK_DIM = 2;
constexpr int N = THREAD_DIM * BLOCK_DIM;

__device__ float max(float a) {
	if (a > 0) 
        	return a;
	else
	     return 0;
}

__device__ float min(float a) {
	if (a < 0) 
		return a;
	else 
		return 0;
}	

__global__ void relu_kernel(float* A, float* B, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		B[idx] = max(A[idx]);
	}
}

__global__ void leaky_relu_kernel(float* A, float* B, float alpha, int N) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < N) {
		B[idx] = max(A[idx]) + alpha * min(A[idx]);
	}
}


__global__ void swish_kernel(float* A, float* B, float beta, int N) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		B[idx] = A[idx] / (1.0f+expf(-beta * A[idx]));
	}

}

void SwishInit() {
    // Host allocations
    float* A_h = new float[N];
    float* B_h = new float[N];

    const float beta = 1.0f;
    const float alpha = -0.9f;

    // Initialize A_h with some data
    for (int i = 0; i < N; ++i) {
        A_h[i] = static_cast<float>(i) / N;  // [0.0, 0.001, ..., 1.0]
    }

    // Device allocations
    float *A_d, *B_d;
    cudaMalloc(&A_d, N * sizeof(float));
    cudaMalloc(&B_d, N * sizeof(float));

    // Copy input to device
    cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch
    int threadsPerBlock = THREAD_DIM;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, alpha, N);

    // Copy output to host
    cudaMemcpy(B_h, B_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first 10 results for sanity
    for (int i = 0; i < 10; ++i) {
        std::cout << "A[" << i << "] = " << A_h[i]
                  << ", Swish = " << B_h[i] << std::endl;
    }

    // Cleanup
    delete[] A_h;
    delete[] B_h;
    cudaFree(A_d);
    cudaFree(B_d);
}

int main() {
	SwishInit();
}
