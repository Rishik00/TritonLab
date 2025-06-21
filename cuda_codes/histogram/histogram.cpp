#include <iostream>
#include <string>

void histogram_sequential(char *data, unsigned int length, unsigned int *histo, unsigned int numBuckets) {
    std::cout << "Input data: ";
    for (unsigned int i = 0; i < length; i++) {
        std::cout << data[i];
    }
    std::cout << std::endl;

    for (unsigned int i = 0; i < length; i++) {
        int pos = data[i] - 'a';
        std::cout << "Processing character '" << data[i] << "' at index " << i 
                  << " → pos = " << pos << std::endl;

        if (pos >= 0 && pos < 26) {
            unsigned int bucket = pos / numBuckets;
            std::cout << "  Valid letter, bucket = " << bucket << std::endl;
            histo[bucket]++;
        } else {
            std::cout << "  Skipped (not a lowercase a-z letter)" << std::endl;
        }
    }
}

int main() {
    std::string data = "chromium";
    unsigned int numBuckets = 4;
    unsigned int *histo = new unsigned int[numBuckets]();  // initialize to zero

    histogram_sequential(&data[0], data.length(), histo, numBuckets);

    std::cout << "\nHistogram result:\n";
    for (unsigned int i = 0; i < numBuckets; i++) {
        std::cout << "Bucket " << i << ": " << histo[i] << std::endl;
    }

    delete[] histo;
    return 0;
}
