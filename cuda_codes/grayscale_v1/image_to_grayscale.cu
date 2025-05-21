#include <torch/extension.h>
#include <stdio.h>
#include "file.h"
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

// ok idk why it was inline function rather than a normal function? 
// todo:ans - 
unsigned int cdiv(unsigned int a, unsigned int b) {
    // tf is 
    return (a + b - 1) / b;
}

// global is there because this is a global variable 
__global__ void rgb_to_grayscale_kernel(unsigned char* inp, unsigned char* out, int n) {
    // checking idx 
    int idx = blockIdx.x * blockDim + threadIdx.x;
    
    // perform operations on the idx
    if (idx < n) {
        out[idx] = 0.2989*x[idx] + 0.5870*x[idx + n] + 0.1140*x[idx + 2*n];
    }
}

torch::Tensor rgb_to_grayscale(torch::Tensor input) {
    // checking does - checking whether the input is on the device 
    // and checking whether its contiguous in memory or not
    CHECK_INPUT(input);
    
    // Height and width, apparently this is 1 indexed (kill me pls)
    int height = input.size(1);
    int width = input.size(2);

    printf("height*width: %d*%d\n", height, width);

    //init output - eveen i can understand this
    auto output = torch::empty({height, width}, input.options());
    int threads = 256;
`   
    // Kernel launchedn ig? 
    // wonder why the data pointers are unsigned char 
    rgb_to_grayscale_kernel<<<cdiv(height, width), threads>>>(
        input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>()
    );

    // Kernel launch being checked ig?
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

