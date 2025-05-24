// have to rest this

#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <utility>
#include <math.h>

#define VECTOR_SIZE 8

// yet to write this here
float* apply_sigmoid() {
    float* input = new float[VECTOR_SIZE];
    float* output = new float[VECTOR_SIZE];

    for (int i=0; i < N; i++) {
        input[i] = 0.87*i;

        std::cout << input[i] << std::endl;
    }

    for (int i=0; i < N; i++) {
        output[i] = 1 / 1 + ( expf(-input[i]) );

        std::cout << output[i] << std::endl;        
    }

}

int main() {

    return 0;
}