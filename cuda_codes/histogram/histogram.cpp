#include <iostream>
#include <string>

// Fn defn - histo is an array pointer that stores 26 positions 
// or how many ever we want
void histogram_sequential(char *data, unsigned int length, unsigned int *histo, unsigned int numBuckets) {

    if (numBuckets > length){
	    std::cout << "Why do you have number of buckets more than the actual fucking string?" << "\n";
    }

    // Input data printing
    std::cout << "Input data: ";
    for (unsigned int i = 0; i < length; i++) {
        std::cout << data[i];
    }
    std::cout << std::endl;

    // For each charecter, do the following: 
    // position = data[i] - 'a'
    // histo[position/numBuckets] = histo[position/numBuckets] + 1
    for (unsigned int i = 0; i < length; i++) {

	int pos = data[i] - 'a';
        std::cout << "Processing character '" << data[i] << "' at index " << i 
                  << "pos = " << pos << std::endl;

        if (pos >= 0 && pos < 26) {
            unsigned int bucket = pos / numBuckets;
            histo[bucket]++;
        } else {
            std::cout << "  Skipped (not a lowercase a-z letter)" << std::endl;
        }
    }
}

int main() {
    std::string data = "chromium";
    // Apparently std::string has a fn length() 
    
    unsigned int numBuckets = 4; // Can be anything, but can't be more than 26 because just wastes memory
				 // [TODO] Add some fucking checks
    unsigned int *histo = new unsigned int[numBuckets]();  // initialize to zero
    histogram_sequential(&data[0], data.length(), histo, numBuckets);

    std::cout << "\nHistogram result:\n";
    for (unsigned int i = 0; i < numBuckets; i++) {
        std::cout << "Bucket " << i << ": " << histo[i] << std::endl;
    }

    delete[] histo;
    return 0;
}
