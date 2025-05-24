#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>
#include <math.h>


unsigned int ceilingDiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__ void vector_sigmoid_kernel(float* M, float* P, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        P[idx] = 1 / (1 + expf(x));
    }
}

std::pair<float*, float*> init_vector(int size) {

    float* input_vector =  new float[size];
    float* output_vector =  new float[size];

    // shoving some values
    for (int i=0; i < size; i++) {
        input_vector[i] = 0.87*i;
    }

    std::cout << "Input: " <<std::endl;
    for(int j=0;j<size;j++) {
        std::cout << input_vector[j] << ' ';
    }
    std::cout << std::endl;

    return {input_vector, output_vector};
}

float* vector_sigmoid(int size) {
    auto [Mh, Nh] = init_vector(size);

    float* Md, *Nd;

    cudaMalloc(&Md, size * sizeof(float));
    cudaMalloc(&Nd, size * sizeof(float));

    cudaMemcpy(Md, Mh, size * sizeof(float), cudaMemcpyHostToDevice);   
    
    dim3 threadsPerBlock(32);
    dim3 blocksInGrid(ceilingDiv(size, threadsPerBlock.x));

    vector_sigmoid_kernel<<<blocksInGrid, threadsPerBlock>>>(Md, Nd, size);
    cudaDeviceSynchronize();

    cudaMemcpy(Nh, Nd, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);

    delete[] Mh;

    return Nh;
}


int main() {
    int size = 64;
    float* output_arr = vector_sigmoid(size);

    std::cout << "Output: " <<std::endl;
    for(int j=0;j<size;j++) {
        std::cout << output_arr[j] << ' ';
    }
    std::cout << std::endl;

    delete[] output_arr;
    return 0;
}