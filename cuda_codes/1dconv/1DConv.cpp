#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <utility>

float* oned_conv(float* A, float* B, float* C, size_t K, size_t N) {

}

std::pair <float*, float*> init_arrays () {
    float* A = new float[5]{3, 4, 5, 7, 9};
    float* B = new float[3]{1, 2, 1};

    return {A, B};  // Return a pair
}
