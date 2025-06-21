#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <utility>

// Histograms length and the vector length
// should be the same?
#define NUM_BINS 26

// Just be sure to change this everytime, else just put it in a fn.
#define STRING_LENGTH 10

__global__ void ParallelHistogram(char *data, unsigned int length, unsigned int *histo, unsigned int numBuckets)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < length)
    {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26)
        {
            atomicAdd(&histo[pos / numBuckets], 1);
        }
    }
}

// Writing directly to global memory. This is fine but has its side effects
__global__ void privateParallelHistogram(char *data, unsigned int length, unsigned int *histo, unsigned int nBuckets)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length)
    {
        unsigned int pos = data[idx] - 'a';
        if (pos >= 0 && pos < 26)
        {
            atomicAdd(&histo[blockIdx.x * NUM_BINS + pos / nBuckets], 1);
        }
    }

    if (blockIdx.x > 0)
    {
        for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockIdx.x)
        {
            unsigned int binValue = histo[blockIdx.x * NUM_BINS + bin];
            if (binValue > 0)
            {
                atomicAdd(&histo[binValue], 1);
            }
        }
    }
}

__global__ void sharedPrivateHistogram(
    char *data,
    unsigned int length,
    unsigned int *histo,
    unsigned int nBuckets)
{
    __shared__ int histo_s[NUM_BINS];
    for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
    {
        histo_s[bin] = 0u;
    }

    __syncthreads();
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < length)
    {
        int pos = data[idx] - 'a';
        if (pos >= 0 && pos < 26)
        {
            atomicAdd(&(histo_s[pos]), 1);
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < NUM_BINS; i += blockDim.x)
    {
        unsigned int binValue = histo_s[i];
        if (binValue > 0)
        {
            atomicAdd(&(histo[i]), binValue);
        }
    }
}

void HistoInit()
{
    const char *data = "rishikesh";
    unsigned int *histo = new unsigned int[NUM_BINS];

    char *data_d;
    unsigned int *histo_d;

    std::cout << "Allocating" << std::endl;
    cudaMalloc((void **)&data_d, STRING_LENGTH * sizeof(char));
    cudaMalloc((void **)&histo_d, NUM_BINS * sizeof(unsigned int));

    std::cout << " Done Allocating, now copying" << std::endl;
    cudaMemcpy(data_d, data, STRING_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    std::cout << " Done Copying , now setting config params" << std::endl;

    unsigned int numBlocks = 2;
    const unsigned int numThreads = (STRING_LENGTH / numBlocks) + 1;

    std::cout << "Kernel time" << std::endl;
    sharedPrivateHistogram<<<numBlocks, numThreads>>>(data_d, STRING_LENGTH, histo_d, NUM_BINS);

    cudaDeviceSynchronize();

    cudaMemcpy(histo, histo_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < NUM_BINS; i++)
    {
        std::cout << "Bin " << i << ": " << histo[i] << std::endl;
    }

    cudaFree(data_d);
    cudaFree(histo_d);

    delete[] histo;
}

int main()
{
    HistoInit();
}
