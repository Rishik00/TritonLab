#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <utility>

std::pair<float*, float*> init_vector(int size) {
    int total_size = size * size;
    float* P = new float[total_size];
    float* R = new float[total_size];

    for(int i=0; i<total_size; ++i) {
        P[i] = static_cast<float>(i); // Explicitly cast to float
    }

    std::cout << "Source array: " << std::endl;
    for (int i=0; i<size;i++){
        for(int j=0;j<size;j++) {
            std::cout << P[i * size + j] << ' ';
        }
        std::cout << std::endl;

    }

    return {P, R};
}

void transpose_2d_matrix(int size) {
    auto [M, R] = init_vector(size);

    for (int i=0; i<size;i++){
        for(int j=0;j<size;j++) {
            R[i * size + j] = M[j * size + i];
        }
    }

    std::cout << "Transposed array: " << std::endl;
    for (int i=0; i<size;i++){
        for(int j=0;j<size;j++) {
            std::cout << R[i * size + j] << ' ';
        }
        std::cout << std::endl;

    }

    delete[] R;
    delete[] M;
}


int main() {
    int size = 4;
    transpose_2d_matrix(size);
}