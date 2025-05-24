#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>

unsigned int ceilingDiv (unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}


// ELEMENT WISE TRANSPOSE, i.e - output[i][j] = input[j][i]
__global__ void blockTranspose(float* Ad, float* Bd, int Aw, int Ah) {
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int row = bidy * blockDim.y + tidy;
    int col = bidx * blockDim.x + tidx;

    if (row < Ah && col < Aw) {
        // Calculate linear index for the input array (Ad)
        int in_idx = row * Aw + col;

        // Calculate linear index for the output (transposed) array (Bd)
        int out_idx = col * Ah + row;

        printf("Thread (%d, %d) copying A[%d] = %.2f to B[%d]\n",
                            row, col, in_idx, Ad[in_idx], out_idx);

        Bd[out_idx] = Ad[in_idx];
    }
}

float* init_vector(int size) {
    int total_size = size * size;
    float* P = new float[total_size];

    for(int i=0; i<total_size; ++i) {
        P[i] = static_cast<float>(i); // Explicitly cast to float
    }

    std::cout << size << std::endl;
    for (int i=0; i<size;i++){
        for(int j=0;j<size;j++) {
            std::cout <<    P[i * size + j] << ' ';
        }
        std::cout << std::endl;

    }

    return P;
}

float* block_transpose(int size) {
    int total_size = size * size;

    auto Mh = init_vector(size);
    float *Ph = new float[total_size];
    float* Md, *Pd;

    cudaMalloc(&Md, total_size * sizeof(float));
    cudaMalloc(&Pd, total_size * sizeof(float));

    cudaMemcpy(Md, Mh, total_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2, 2); 
    dim3 blocksInGrid(ceilingDiv(size, threadsPerBlock.x),
                    ceilingDiv(size, threadsPerBlock.y));
    
    blockTranspose<<<blocksInGrid, threadsPerBlock>>>(Md, Pd, size, size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(Ph, Pd, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Pd);
    delete[] Mh;

    return Ph;
}


int main() {
    int size = 2;
    float* output_arr = block_transpose(size);

    for (int i=0; i<size;i++){
        for(int j=0;j<size;j++) {
            std::cout << output_arr[i * size + j] << ' ';
        }
        std::cout << std::endl;

    }
    delete[] output_arr;
    return 0;
}