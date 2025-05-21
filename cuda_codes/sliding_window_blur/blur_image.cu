#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>

unsigned int ceilingDiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__ void blur_image_kernel(
    float* input, 
    float* output, 
    int width, 
    int height) {

    int di;
    int dj;
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if (tidx >= width || tidy >= height) return;
    float sum = 0.0f;

    for (di = -1; di <= 1; di++) {
        for (dj = -1; dj <= 1; dj++) {

            if (tidx < width && tidy < height) {
                int new_i = tidx + di;
                int new_j = tidy + dj;

                sum += input[new_i * width + new_j];
            }
            
        }
    }
    output[tidx * width + tidy] = sum / 9.0f;
}

float* blur_image(
    int width,
    int height,
    int size, 
    float value) {

    int size = width * height;
    auto [h_input_ptr, h_output_ptr] = init_arrays_host(size, value);
    int n_threads = 256;

    float* d_input_ptr, d_output_ptr;
    cudaMalloc(&d_input_ptr, size * sizeof(float));
    cudaMalloc(&d_output_ptr, size * sizeof(float));

    cudaMemcpy(d_input_ptr, h_input_ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    // if width = 4096 and height  4096, then cdiv will give 
                // (4096 * 4096 + 256) - 1 / 256

    // alright, init for dimblocks and dimthreads
    dim3 threadsDim(16, 16);
    dim3 blockDim(ceilingDiv(width, threadsDim.x), ceilingDiv(height, threadsDim.y));

    // wtf is this? 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    blur_image_kernel<<<blockDim, threadsDim>>>(d_input_ptr, d_output_ptr, width, height);
    cudaMemcpy(h_output_ptr, d_output_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);

    // excuse me bruv? 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);  // Get elapsed time in ms
    std::cout << "Kernel Execution Time: " << milliseconds << " ms\n";

    cudaFree(d_input_ptr);
    cudaFree(d_output_ptr);

    delete[] h_input_ptr;
    return h_output_ptr;
}

std::pair<float*, float*> init_arrays_host (
    int size,  
    float value) {
    
    float* input_arr =  new float[size];
    float* output_arr =  new float[size];

    if (input_arr == nullptr || output_arr == nullptr) {
        std::cerr << "Memory allocation failed\n";
        exit(1);
    }
    // shoving some values
    for (int i=0; i < size; i++) {
        input_arr[i] = value / 256.0f;
    }

    return {input_arr, output_arr};  // Return a pair
}

int main() {
    int width = 4096;
    int height = 4096;
    float value = 42.0f;
    int size = width * height;

    float* output_arr = blur_image(
        width,
        height, 
        size,
        value
    );

    delete[] output_arr;
}