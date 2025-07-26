## Simple vector addition kernel - Hello world to CUDA!

### Intro
We have two vectors A and B and add each element to a new vector C. Equation is

$$C_i = A_i + B_i$$

### Kernel structure
Each kernel has two components: 

1. The kernel itself thats written using __global__ void vecAdd(float* A, float* B, float* C). 
2. The host function that does the memallocs and memcpys

### Host structure
Host fn has the following: 
1. cudaMalloc - that takes the address of the pointer and the size of the allocation
2. cudaMemcpy copies memory from source to destination. Has two modes:
a. HostToDevice
b. DeviceToHost
3. Kernel launch with 2 params: gridDim (number of blocks in grid) and blockDim (threads per block). 

