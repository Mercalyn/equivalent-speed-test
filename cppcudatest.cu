#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>


/*
speed test c++ cuda
see http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
qty 2 prefilled array size [12k, 12k] multiplying into the first,
20 iterations
consistent result: 240 240 240 240 240
*/


#define SIZE_X_ARRAY 12000
#define SIZE_Y_ARRAY 12000
#define NUM_ITERATIONS 20
#define FLAT_SIZE (SIZE_X_ARRAY * SIZE_Y_ARRAY)


// timer init
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


// unified mem alloc, this replaces cudaMallocManaged
__device__ __managed__ double aArr[FLAT_SIZE];
__device__ __managed__ double bArr[FLAT_SIZE];


__global__ void prefillKernels() {
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;
    int gid = (SIZE_X_ARRAY * y) + x;

    // bounds check
    if (gid < FLAT_SIZE) {
        // random setup
        //curandState_t state;
        curandStatePhilox4_32_10_t state; // philox pseudorandom is much faster
        curand_init(50227, gid, 0, &state);

        // return random double between 0-2
        aArr[gid] = (curand_uniform(&state) * 2.0);
        bArr[gid] = (curand_uniform(&state) * 2.0);
    }
}


__global__ void multKernel() {
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;
    int gid = (SIZE_X_ARRAY * y) + x;

    // bounds check
    if (gid < FLAT_SIZE) {
        // return elementwise mult into a
        aArr[gid] *= bArr[gid];
    }
}


void debugPrint() {
    int gid;

    // a
    std::cout << "a: \n";
    for (int y = 0; y < SIZE_Y_ARRAY; y++) {
        for (int x = 0; x < SIZE_X_ARRAY; x++) {
            gid = (y * SIZE_X_ARRAY) + x;
            std::cout << aArr[gid] << " -- ";
        }
        std::cout << "\n";
    }
    std::cout << "----------------------------\nb: \n";

    // b
    for (int y = 0; y < SIZE_Y_ARRAY; y++) {
        for (int x = 0; x < SIZE_X_ARRAY; x++) {
            gid = (y * SIZE_X_ARRAY) + x;
            std::cout << bArr[gid] << " -- ";
        }
        std::cout << "\n";
    }
    std::cout << "----------------------------\n";
}


void debugPrintLast() {
    int lastIndex = (static_cast<int>(FLAT_SIZE)) - 1;

    std::cout << "lastIndex a: \n";
    std::cout << aArr[lastIndex];
    std::cout << "\nlastIndex b: \n";
    std::cout << bArr[lastIndex];
    std::cout << "\n----------------------------\n";
}


int main() {
    // choosing device is unnecessary
    //cudaSetDevice(0);
    std::cout << std::setprecision(15); // debug::actual double values?


    // dim and sizes
    int blockCeil[2] = { 
        static_cast<int> ((SIZE_Y_ARRAY / 32) + 1),
        static_cast<int> ((SIZE_X_ARRAY / 32) + 1)
    };
    //std::cout << blockCeil[0] << " " << blockCeil[1] << "\n";
    dim3 numBlocks(blockCeil[0], blockCeil[1]); // thanks to 2d limit of 32, 32 in threads, we will grab ceiling of each dimension / 32
    dim3 numThreads(32, 32); // 1024 is max in its own flat dimension, it isn't accepting above 32, 32, see https://forums.developer.nvidia.com/t/maximum-number-of-threads-on-thread-block/46392

    // prefilling both happens on a single kernel since unified mem is global
    std::cout << "prefilling... \n\n";
    prefillKernels <<< numBlocks, numThreads >>> ();
    cudaDeviceSynchronize();
    //debugPrint();
    debugPrintLast();

    // process
    std::cout << "processing... \n\n";
    auto t1 = high_resolution_clock::now(); // start timer
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        multKernel <<< numBlocks, numThreads >>> (); // <<< num of blocks desired, number of threads not to exceed 1024 >>>
    }

    // check err
    cudaGetLastError();
    // sync // aka join
    cudaDeviceSynchronize();
    //debugPrint();
    debugPrintLast();

    // stop timer, log
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << ms_int.count() << "ms\n";

    cudaFree(aArr);
    cudaFree(bArr);
    return 0;
}

