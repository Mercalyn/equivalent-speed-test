#include <iostream>
#include <chrono>
#include <random>
#include <thread>

/*
speed test using multithreading
qty 2 prefilled array size [12k, 12k] multiplying into the first,
10 iterations(threaded)
consistent result: 875ms -- WOW!
*/

int const SIZE_X_ARRAY = 12000;
int const SIZE_Y_ARRAY = 12000;
int const NUM_ITERATIONS = 10;

// seed random from c++11 random
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(.0001, 2.000);


// run ea process as a thread
void process(double* aPtrs[], double* bPtrs[]){
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            aPtrs[y][x] *= bPtrs[y][x];
        }
    }
}

// might as well prefill as thread as well
void prefill(double* arrayPtrs[]){
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            //printf("%p\n", &(arrayPtrs[y][x])); // retrieve the address [y][x]
            arrayPtrs[y][x] = dist(mt);
        }
    }
}

int main(){
    // qty: 2, 2d arrays (pointer arrays of pointers)
    double* aPtrs[SIZE_Y_ARRAY];
    double* bPtrs[SIZE_Y_ARRAY];
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        aPtrs[y] = new double[SIZE_X_ARRAY];
        bPtrs[y] = new double[SIZE_X_ARRAY];
    }
    // array of threads (only for num iterations and processing threads)
    std::thread processThreads[NUM_ITERATIONS];
    
    
    // prefill threads (separate from above)
    std::thread aPre(prefill, aPtrs);
    std::thread bPre(prefill, bPtrs);
    // wait on both prefills
    aPre.join();
    bPre.join();
    std::cout << "done prefilling\n";


    // timer init
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();


    // process the work using threads
    for(int i = 0; i < NUM_ITERATIONS; i++){
        //process(aPtrs, bPtrs); // old non multi threaded
        processThreads[i] = std::thread(process, aPtrs, bPtrs);
    }
    // wait for all threads
    for(int i = 0; i < NUM_ITERATIONS; i++){
        processThreads[i].join(); // i cannot believe this works
    }
    
    
    //stop timer and print out
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