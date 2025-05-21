#include <iostream>
#include <vector>
#include <utility>


float* Matmul(int size, float value) {
    float *M, *N = initInputMats(size, value);
    float *P = initOutputMat(size);    
    
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++){
            for (k=0; k<size;k++) {
                P[i * size + j] += M[i * size + k] * N[size * k + j];
            }
        }
    }
    
    delete[] M; 
    delete[] N;
    return P;
}

std::pair<float*, float*> initInputMats(int size, int value) {
    int total_size = size * size;
    float* M = new float[total_size];
    float* N = new float[total_size];

    if (input_arr == nullptr || output_arr == nullptr) {
        std::cerr << "Memory allocation failed\n";
        exit(1);
    }
    // shoving some values
    for (int i=0; i < size; i++) {
        M[i] = value / 256.0f;
        N[i] = (value + value) / 256.0f;
    }

    return {M, N};
}

float* initOutputMat(int size) {
    int total_size = size * size;
    float* P = new float[total_size];
}

int main () {
    int size=14;
    float value=42.0f; 
 
    float* P = Matmul(size, value);
    std::cout << "Got it!" << endl;
    delete[] P;
    return 0;    
}