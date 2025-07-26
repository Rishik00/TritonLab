## Matmul kernel - the kernel that made me hate my life

We can consider this to be the successor to vecAdd. But don't be deceived by its simplicity, a lot of work goes into optimiziing this kernel. 

## Op - naive 
Matrix multiplication has 3 loops. Two of them are for iterating through the rows and columns. The last loop is for indexing each row and column to perform the dot product. 

\[
C_{i,j} = \sum_k A_{i,k} \cdot B_{k,j}
\]

So, the dot product can be characterized in 2 dimensions: 
1. Get the row and column index for each thread. 
2. Perform the final dot product loop for each of these. 


## Op - tiled 
The naive version works, but isn't optimized enough for speed due to 2 reasons: 
1. Memory badndwidth, because you're doing a lot of reads and writes to global memory which is a very low bandwidth high volume memory structure. 

To solve this, we use shared memory, think of it as a much faster cache that is low in size but has extremely low load times; which is nice for us because each thread can then focus on performing the op than actually waiting for stuff to happen. 

But there's a problem. Shared memory is very limited in size compared to global memory. So we have to split each of our blocks into individual tiles and load those into shared memory; one at a time and perform the op. 

```#define TILE_SIZE 16

__global__ void matMulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}```

