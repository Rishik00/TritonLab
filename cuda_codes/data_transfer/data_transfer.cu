#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>

constexpr int THREADS_PER_BLOCK = 1024;
constexpr int BLOCKS_IN_GRID = 1;
constexpr int VECTOR_DIM = THREADS_PER_BLOCK * BLOCKS_IN_GRID;

unsigned int ceilingDiv (unsigned int a, unsigned int b) {
	return (a + b - 1) / b;
}

__global__ void simpleDataTransfer(float* A, float* B, int N) {
	int idx = threadIdx.x + blockIdx.x  * blockDim.x;
	printf("BlockIdx: %d, threadIdx: %d\n", blockIdx.x, threadIdx.x);

	if (idx < N) {
		B[idx] = A[idx];
	}
}

void DataTransferInit() {
	float* A_h = new float[VECTOR_DIM];
	float* B_h = new float[VECTOR_DIM];	
	for (int i = 0; i < VECTOR_DIM; ++i) {
	    A_h[i] = 1.0f;
	}

	float *A_d, *B_d;
	cudaMalloc(&A_d, sizeof(float) * VECTOR_DIM);
	cudaMalloc(&B_d, sizeof(float) * VECTOR_DIM);

	cudaMemcpy(A_d, A_h, VECTOR_DIM * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blockDim(THREADS_PER_BLOCK);               // Threads per block
	dim3 gridDim(ceilingDiv(VECTOR_DIM, blockDim.x)); // Number of blocks

	simpleDataTransfer<<<gridDim, blockDim>>>(A_d, B_d, VECTOR_DIM);
	cudaDeviceSynchronize();

	cudaMemcpy(B_h, B_d, sizeof(float) * VECTOR_DIM, cudaMemcpyDeviceToHost);

	int count = 0;
	for  (int i=0; i < VECTOR_DIM; i++) {
			count+=1;
	}
	std::cout << count << " " << VECTOR_DIM << std::endl;
	if (count == VECTOR_DIM) {
		std::cout << "What?" << std::endl;
	}

	std::cout << "Done?" << std::endl;

	cudaFree(B_d);
	cudaFree(A_d);

	delete[] A_h;
	delete[] B_h;
}

int main() {
	DataTransferInit();
}
