#include <iostream>
#include <string>
#include <cuda_runtime.h>

// For pseduo random generation
#include <curand.h>

constexpr int N = 32;
constexpr int HIDDEN_DIM = 1;
constexpr int NUM_THREADS = N;

unsigned int ceilingDiv(unsigned int a, unsigned int b) {
	return (a + b - 1) / b;
}

__device__ float softmax(float* in, float* out, float* max, float* sum) {
	
}

__device__ float ReLU(float num) {
	if (num > 0) {
		return num;
	} else {
		return 0;
	}
}

__device__ float sigmoid(float num) {
	// TODO: Write sigmoid later
}

__device__ float boxMuller(float u1, float u2) {
	// TODO: implement this	
}

// Define matmul here using tiling - this would be the entry point and would have two versions: ReLU and sigmoid
__global__ void forwardKernelwithReLU(float* weight, float* bias, float* input, float* out){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// TODO: scaling using boxMuller transform
	if (idx < N) {
		float Aval = 0.0f;

		for (int i = 0; i < N; i++) {
			int index = idx * N + i;
			Aval += input[index] * weight[idx];
		}
		__syncthreads();

		out[idx] = ReLU(Aval + bias[idx]);
	} 
	__syncthreads();

}


// host fns
void LaunchNN(const char* activation) {
	float* o_h = new float[N];
	float* outData = new float[N];

	for (int i = 0; i < N; i++) {
		outData[i] = 0.0f;
	}

	curandGenerator_t param_gen;
	curandGenerator_t input_gen;

	float *weightData, *bias;
	float *inputData;

	cudaMalloc(&weightData, sizeof(float) * N * HIDDEN_DIM);
	cudaMalloc(&bias, sizeof(float) * N);
	cudaMalloc(&inputData, sizeof(float) * N);
	cudaMalloc(&outData, sizeof(float) * N);

	// This is going to create the random number generator - for weight and input init
	curandCreateGenerator(&param_gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandCreateGenerator(&input_gen, CURAND_RNG_PSEUDO_DEFAULT);

	// Set seed - 1234 (unsigned long int) for weights
	curandSetPseudoRandomGeneratorSeed(param_gen, 1234ULL);
	curandSetPseudoRandomGeneratorSeed(input_gen, 4567ULL);

	// Set the seed - 4567 for inputs
	curandGenerateUniform(input_gen, inputData, N*N);
		
	// Generate n floats on device - params
	curandGenerateUniform(param_gen, weightData, N*HIDDEN_DIM);
	curandGenerateUniform(param_gen, bias, N);

	// Kernel launch
	dim3 blockDim(NUM_THREADS, NUM_THREADS);
	dim3 gridDim(1,1);

	forwardKernelwithReLU<<<gridDim, blockDim>>>(weightData, bias, inputData, outData);

	cudaMemcpy(outData, o_h, N * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Freeing the params
	cudaFree(weightData);
	cudaFree(bias);

	std::cout << "Done with this "<< "\n";
}

int main(int argc, char* argv[]) {
	const char* activation = "relu";
	std::cout << argc << "\n";
	
	if (argc < 2) {
		std::cout << "No activation porvided in args, Defaulting to " << activation << "\n";
	} else {
		activation = argv[1];
		std::cout << "Activation selected: " << activation << "\n";
	}

	LaunchNN(activation);
}
