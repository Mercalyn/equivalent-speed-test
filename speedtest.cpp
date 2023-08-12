#include <iostream>
#include <chrono>
#include <random>

/*
speed test
qty 2 prefilled array size [12k, 12k] multiplying into the first,
20 iterations
consistent result: 5480 5760 5950 6000 6160
*/

int const SIZE_X_ARRAY = 12000;
int const SIZE_Y_ARRAY = 12000;
int const NUM_ITERATIONS = 10;

// seed random from c++11 random
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(.0001, 2.000);


// run this process for num iterations
void process(double* aPtrs[], double* bPtrs[]){
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            aPtrs[y][x] *= bPtrs[y][x];
        }
    }
}

void prefill(double* arrayPtrs[]){
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            //printf("%p\n", &(arrayPtrs[y][x])); // retrieve the address [y][x]
            arrayPtrs[y][x] = dist(mt);
        }
    }
}

int main(){
    // 2 new pointer arrays of pointers
    double* aPtrs[SIZE_Y_ARRAY];
    double* bPtrs[SIZE_Y_ARRAY];
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        aPtrs[y] = new double[SIZE_X_ARRAY];
        bPtrs[y] = new double[SIZE_X_ARRAY];
    }
    
    // prefill
    prefill(aPtrs);
    prefill(bPtrs);

    // timer init
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    // start timer and loop
    auto t1 = high_resolution_clock::now();
    for(int i = 0; i < NUM_ITERATIONS; i++){
        process(aPtrs, bPtrs);
    }
    
    // get end timer and print out
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << ms_int.count() << "ms\n";

    // debug test if it actually did the work
    /*
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            printf("%.8f\n", aPtrs[y][x]);
        }
    }
    */

}