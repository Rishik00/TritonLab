#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

int CHANNELS 3;

unsigned int ceilingDiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__ void image_to_grayscale_kernel(float* input_matrix, float* output_matrix, int m, int n) {
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row_idx = threadIdx.y + blockIdx.y * blockDim.y;

    if (col_idx < n && row_idx < m) {
        //core logic

        int offset_idx = row_idx * m + col_idx;

        int offset = offset_idx * CHANNELS;
        float r_offset = input_matrix[offset];
        float g_offset = input_matrix[offset + 1];
        float b_offset = input_matrix[offset + 2];

        output_matrix[offset_idx] - 0.21 * r_offset + 0.71 * g_offset + 0.17 * b_offset;
    }
}


std::pair<float*, float*> init(int size) {
    int total_size = size * size;

    float* input_matrix =  new float[total_size];
    float* output_matrix =  new float[total_size];

    // shoving some values
    for (int i=0; i < total_size; i++) {
        input_matrix[i] = 10;
    }

    return {input_matrix, output_matrix};  // Return a pair    
}

void image_to_grayscale(int size) {
    int n_threads = 16;
    int total_size = size * size;

    auto [h_input_matrix, h_output_matrix] = init(size);
    float* d_input_matrix, *d_output_matrix;

    cudaMalloc(&d_input_matrix, total_size * sizeof(float));
    cudaMalloc(&d_output_matrix, total_size * sizeof(float));

    cudaMemcpy(d_input_matrix, h_input_matrix, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_matrix, h_output_matrix, total_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsDim((n_threads, n_threads);
    dim3 gridDim(ceilingDiv(size, threadsDim.x), ceilingDiv(size, threadsDim.y));

    // should be 
    // <<< gridDim --> Number of blocks in the grid >>>
    // <<< blockDim --> Number of threads in a block >>>
    image_to_grayscale_kernel<<<gridDim, threadsDim>>>(d_output_matrix, d_input_matrix, size, size);
    cudaMemcpy(h_output_matrix, d_output_matrix, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input_matrix);
    cudaFree(d_output_matrix);
    
    std::cout << "Entering the after free phase" << std::endl;
    delete[] h_input_matrix;

    for (int i=0; i<size;i++) {
        std::cout << "Element: " << h_output_matrix[i] << std::endl;
    }
    delete[] h_input_matrix;
}


int main() {
    int size = 5;
    image_to_grayscale(size);
}