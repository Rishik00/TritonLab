#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>

__global__ void oned_conv_kernel(float* A, float* B, float* C, size_t K, size_t N) {
    int idx = threadIdx.x;
    float res = 0.0f;

    for (int j = 0; j < K; j++ ){
        res += A[idx+j] * B[j];
    }
    C[idx] = res;
}

std::pair <float*, float*> init_arrays () {
    float* A = new float[5]{3, 4, 5, 7, 9};
    float* B = new float[3]{1, 2, 1};

    return {A, B};  // Return a pair
}

float* oned_conv() {
    auto [A_h, B_h] = init_arrays();
    float* C_h = new float[5];

    int N = 5;
    int K = 3;
    int n_threads = 8;

    float* A_d;
    float* B_d;
    float* C_d;
    cudaMalloc(&A_d, N * sizeof(float));
    cudaMalloc(&B_d, K * sizeof(float));
    cudaMalloc(&C_d, N * sizeof(float));

    cudaMemcpy(A_d, A_h, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size * sizeof(float), cudaMemcpyHostToDevice);

    oned_conv_kernel<<<1, n_threads>>>(A_d, B_d, C_d, N, K);
    cudaMemcpy(C_h, C_d, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    delete[] A_h;
    delete[] B_h;
    return C_h;
}


int main() {
    oned_conv();
}