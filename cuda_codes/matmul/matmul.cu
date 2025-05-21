#include <iostream>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

using namespace std;

unsigned int ceilingDiv (unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

__global__ void matmul_kernel(float* Md, float* Nd, float* Pd, int n) {    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float Pvalue = 0.0f;
        for (int m = 0; m < n; m++) {
            Pvalue += Md[row * n + m] * Nd[m * n + col];
        }
        Pd[row * n + col] = Pvalue;
    }
}

pair<float*, float*> InitHostArrays(int total_size, float value) {
    float* Mh = new float[total_size];
    float* Nh = new float[total_size];

    if (Mh == nullptr || Nh == nullptr) {
        cerr << "Memory allocation failed\n";
        exit(1);
    }

    // Fill arrays with values
    for (int i = 0; i < total_size; i++) {
        Mh[i] = value / 256.0f;
        Nh[i] = (value + value) / 256.0f;
    }

    return {Mh, Nh};
}

float* matmul(int n, float value) {
    int total_size = n * n;

    auto [Mh, Nh] = InitHostArrays(total_size, value);
    float* Ph = new float[total_size];

    float* Md, *Nd, *Pd;
    cudaMalloc(&Md, total_size * sizeof(float));
    cudaMalloc(&Nd, total_size * sizeof(float));
    cudaMalloc(&Pd, total_size * sizeof(float));

    cudaMemcpy(Md, Mh, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, Nh, total_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsDim(32, 32);
    dim3 blocksDim(
        ceilingDiv(n, threadsDim.x), 
        ceilingDiv(n, threadsDim.y)
    ); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_kernel<<<blocksDim, threadsDim>>>(Md, Nd, Pd, n);
    cudaMemcpy(Ph, Pd, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Kernel Execution Time: " << milliseconds << " ms\n";

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);

    delete[] Mh;
    delete[] Nh;

    return Ph;
}

int main() {
    int size = 4096;  // this is 'n', dimension of the matrix
    float value = 42.0f;

    float* output_arr = matmul(size, value);

    delete[] output_arr;
    return 0;
}
