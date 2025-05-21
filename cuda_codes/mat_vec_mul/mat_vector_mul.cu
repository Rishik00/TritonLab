#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

unsigned int ceilingDiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__ void matrix_vector_multiplication_kernel(float* out_d, float* input_matrix_d, float* input_vector_d, int size) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float Avalue = 0.0;
        for (int i=0; i<size;i++) {
            int index = idx * size + i;
            Avalue += input_matrix_d[index] * input_vector_d[i];
        }
        out_d[idx] = Avalue;
    }

}

std::pair<float*, float*> init(int size) {
    int total_size = size * size;

    float* input_matrix =  new float[total_size];
    float* input_vector =  new float[size];

    // shoving some values
    for (int i=0; i < total_size; i++) {
        input_matrix[i] = 10;
    }

    for (int i=0; i < size; i++) {
        input_vector[i] = i;
    }

    return {input_matrix, input_vector};  // Return a pair    
}

void matrix_vector_multiplication(int size) {
    int n_threads = 16;
    int total_size = size * size;

    auto [h_input_matrix, h_input_vector] = init(size);
    float* h_output_vector = new float[size];
    float* d_input_matrix, *d_input_vector, *d_output_vector;

    cudaMalloc(&d_input_matrix, total_size * sizeof(float));
    cudaMalloc(&d_input_vector, size * sizeof(float));
    cudaMalloc(&d_output_vector, size * sizeof(float));

    cudaMemcpy(d_input_matrix, h_input_matrix, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_vector, h_input_vector, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsDim(n_threads);
    dim3 gridDim(ceilingDiv(size, threadsDim.x));

    // should be 
    // <<< gridDim --> Number of blocks in the grid >>>
    // <<< blockDim --> Number of threads in a block >>>
    matrix_vector_multiplication_kernel<<<gridDim, threadsDim>>>(d_output_vector, d_input_matrix, d_input_vector, size);
    cudaMemcpy(h_output_vector, d_output_vector, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input_vector);
    cudaFree(d_input_matrix);
    cudaFree(d_output_vector);
    
    std::cout << "Entering the after free phase" << std::endl;
    delete[] h_input_matrix;
    delete[] h_input_vector;

    for (int i=0; i<size;i++) {
        std::cout << "Element: " << h_output_vector[i] << std::endl;
    }
}

int main() {
    int size = 7;
    matrix_vector_multiplication(size);
}