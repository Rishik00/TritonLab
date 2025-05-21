#include <iostream>
#include <cstdlib>

using namespace std;

float* apply_blurop(float* input_arr, float* output_arr, int width, int height) {

    for (int i=0; i < width; i++) {
        for (int j=0; j < width; j++) {
            float sum  =0.0f;

            for (int di = -1; di <= 1; di ++ ){
                for (int dj = -1; dj <= 1; dj ++ ) {

                    int new_i = i + di;
                    int new_j = j + dj;

                    if(new_i < 0 || new_i > height ||new_j < 0 || new_j >width) {
                        continue;
                    }

                    sum += input_arr[new_i * width + new_j];
                }
            }

            output_arr[i * width + j] = sum / 9.0f;
        }
    }

    return output_arr;
}

float* init_image (int size, int value) {
    // allocating dynamically to the heap
    // arr[i][j] --> arr (i * WIDTH_ARRAY + j)

    float* arr = new float[size];

    // classic nullptr
    if (arr == nullptr) {
        exit(1);
    }

    // shoving some values
    for (int i=0; i < size; i++) {
        arr[i] = value / 256.0f;
    }
    return arr;
}

float* init_output (int size) {
    float* arr = new float[size];
    return arr;
}

int main() {
    int width = 14;
    int height = 14;

    int size = width * height;

    // Initializing image with value 42
    float* my_array = init_image(size,  42.0f);
    float* output_array = init_output(size);

    cout << "Memory allocated" << endl;
    cout << "========================================" << endl;
    cout << "Input array:" << endl;

    // Printing first 5 values of the input array
    for (int i = 0; i < 5; ++i) {
        cout << my_array[i] << " ";
    }
    cout << endl;
    cout << "========================================" << endl;

    // call the fn 
    float* result = apply_blurop(my_array, output_array, width, height);
    cout << "Done with blurring!";

    for (int i = 0; i < 5; ++i) {
        cout << result[i] << " ";
    }
    cout << endl;
    cout << "========================================" << endl;

    // C's version of free
    delete[] my_array;
    delete[] output_array;

    return 0;
}