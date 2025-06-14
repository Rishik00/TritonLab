#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define WIDTH 28
#define HEIGHT 28

unsigned int ceilingDiv (unsigned int a, unsigned int b) {
    //ceiling division function
    return (a + b - 1) / b;
}

__global__ void multiply_random_kernel(float* output, float* input, int n) {
    // check the idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // iterate over the array (or not)
    if (idx < n) {
        // do the op and save to output
        output[idx] = input[idx] * 0.0432;
    }

}

int main() {
    int size = WIDTH * HEIGHT;
    int n_threads = 128;
   
    // allocate the input and output memory for the host device
    float* h_image = (float*)malloc(size * sizeof(float));
    float* h_result = (float*)malloc(size * sizeof(float));

    // Simulate an image with random pixel values [0, 255] - 28x28
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        h_image[i] = rand() % 256;
    }

    // cudamalloc for allocating memory to GPU
    float* d_input, d*output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    // Copying the contents of input from host to device
    cudaMemCpy(d_input, h_image, size * sizeof(float), cudaMemcpyHostToDevice);

    // start the kernel 
    multiple_random_kernel<<<ceilingDiv(size, 256), n_threads>>>(d_input, d_output, size);

    // copy outputs from device to host
    cudaMemCpy(h_result, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Normalized Image:\n");
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            printf("%.2f ", h_result[i * WIDTH + j]);
        }
        printf("\n");
    }

    // clean up
    free(h_image);
    free(h_result);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}