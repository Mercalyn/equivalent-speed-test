#include <iostream>
#include <chrono>

/*
speed test
100k iterations of 1 dimensional array of size 16k all doing the following:
(1.78939423494 * 0.785186454) + (4.113810156 * .500005123)
*/

int const SIZE_X_ARRAY = 16000;
int const NUM_ITERATIONS = 100000;

// run this process for num iterations
void process(int* startArrayPtr){
    for(int x = 0; x < SIZE_X_ARRAY; x++){
        // each item in x array, assign value of pointer to the following math
        *(startArrayPtr + x) = (1.78939423494 * 0.785186454) + (4.113810156 * .500005123);
    }
}

int main(){
    // new array, c version uses malloc but I dont know what .cpp equivalent of that is
    int startArrayPtr[SIZE_X_ARRAY];

    // timer init
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // start timer and loop
    auto t1 = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++){
        process(startArrayPtr);
    }
    
    // get end timer and print out
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << ms_int.count() << "ms\n";

    /*
    // debug test if it actually did the work
    for(int i = 0; i < SIZE_X_ARRAY; i++){
        printf("%d\n", *(startArrayPtr + i));
    }
    */

}