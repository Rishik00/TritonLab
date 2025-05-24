#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>
#include <math.h>

using namespace std;

#define THREADS_PER_BLOCK 4
#define BLOCKS_PER_GRID 1
#define SIZE 8
#define TILE_WIDTH 4
#define VALUE 42.0f

unsigned int ceilingDiv (unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}


__global__ void tiled_matrix_multiplication(float* Md, float* Nd, float* Pd) {
    __shared__ Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ Nds[TILE_WIDTH][TILE_WIDTH];

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int Row = tidy + bidy * TILE_WIDTH;
    int Col = tidx + bidx + TILE_WIDTH;

    float Pval = 0;
    for (int p=0; p < SIZE/TILE_WIDTH; p++) {
        // Loading data to shared memory
        Mds[tidy][tidx] = Md[Row*SIZE + p*TILE_WIDTH + tidx];
        Ndc[tidy][tidx] = Nd[(p * TILE_WIDTH + tidy)*SIZE + Col];
        
        __syncthreads();

        // Operations are the same
        for (int m = 0; m < SIZE; m++) {
            Pval += Mds[SIZE * Row + m] * Nds[m * SIZE + Col];
        }
        Pd[Row * SIZE + Col] = Pval;
        __syncthreads();
    }

}

vector<float*> InitHostArrays(int total_size, float value) {
    float* Mh = new float[total_size];
    float* Nh = new float[total_size];
    float* Ph = new float[total_size];

    if (Mh == nullptr || Nh == nullptr) {
        cerr << "Memory allocation failed\n";
        exit(1);
    }

    // Fill arrays with values
    for (int i = 0; i < total_size; i++) {
        Mh[i] = value / 256.0f;
        Nh[i] = (value + value) / 256.0f;
    }

    return {Mh, Nh, Ph};
}

void apply_tiled_matrix_multiplication() {
    int total_size = SIZE * SIZE;
    auto [Mh, Nh, Ph] = InitHostArrays(total_size, VALUE);
    
    float* Md, *Nd, *Pd;
    cudaMalloc(&Md, total_size * sizeof(float));
    cudaMalloc(&Nd, total_size * sizeof(float));
    cudaMalloc(&Pd, total_size * sizeof(float));

    cudaMemcpy(Md, Mh, total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, Nh, total_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 blocksInGrid(ceilingDiv(SIZE, threadsPerBlock.x),
                      ceilingDiv(SIZE, threadsPerBlock.y));

    tiled_matrix_multiplication<<<blocksInGrid,threadsPerBlock>>>(Md, Nd, Pd, N)
    cudaMemcpy(Ph, Pd, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Md);
    cudaFree(Nd);
    cudaFree(Pd);
   
    delete[] Mh;
    delete[] Nh;
    delete[] Ph;
}

int main() {
    apply_tiled_matrix_multiplication();
    return 0;
}