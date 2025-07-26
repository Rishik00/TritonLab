## Sigmoid - possibly my fav operation.

The operation itself is super simple. Given a vector array we do the following: 

1. Apply exponential to every element and add 1 to it. 
2. Take the inverse of that (1/whatever)

Mathematically: 
[Insert the eqn] 


The convinient thing about this operation is that it can be applied pretty quickly and efficiently. Because it is a vector op, each thread index is an element for the final vector. In CUDA it can be represented by this kernel:  

```
__global__ void vector_sigmoid_kernel(float* M, float* P, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        P[idx] = 1 / (1 + expf(x));
    }
}
```

It's comparable to vecAdd kernel. 

## Improvements
1. Make it for a matrix, and we can apply sigmoid for rows or columns [ONGOING]
2. Make it faster. [LATER]
