#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


/*
speed test
qty 2 prefilled array size [12k, 12k] multiplying into the first,
10 iterations
consistent result: 6.84 secs
*/

int const SIZE_X_ARRAY = 12000;
int const SIZE_Y_ARRAY = 12000;
int const NUM_ITERATIONS = 10;

void process(double* aPtr, double* bPtr){
    for(int x = 0; x < (SIZE_Y_ARRAY * SIZE_Y_ARRAY); x++){
        // each item in x array, mult into a, but cap both ends to avoid exploding and diminishing doubles
        *(aPtr + x) *= *(bPtr + x);
    }
}

void prefill(double* arrayPtr){
    for(int v = 0; v < (SIZE_Y_ARRAY * SIZE_Y_ARRAY); v++){
        // prefill with random values
        *(arrayPtr + v) = (((double)rand()/(double)(RAND_MAX)) * 2) + .00001;
    }
}

int main(){
    // seed random
    time_t t;
    srand((unsigned) time(&t));
    
    // malloc 2 virtual 2d arrays
    double* aPtr = malloc((SIZE_X_ARRAY * SIZE_Y_ARRAY) * sizeof(double));
    double* bPtr = malloc((SIZE_X_ARRAY * SIZE_Y_ARRAY) * sizeof(double));
    
    // prefill 2 array with random values
    prefill(aPtr);
    prefill(bPtr);

    // start timer and loop
    clock_t timer;
    timer = clock();
    for(int i = 0; i < NUM_ITERATIONS; i++){
        process(aPtr, bPtr);
    }
    timer = clock() - timer;

    // get processor time, otherwise it returns 0 sec
    double timeTaken = ((double)timer) / CLOCKS_PER_SEC;
    

    // debug test if it actually did it
    /*
    for(int i = 0; i < (SIZE_Y_ARRAY * SIZE_Y_ARRAY); i++){
        printf("%.8f\n", *(aPtr + i));
    }
    */

    printf("%.3f secs\n", timeTaken);
    
    free(aPtr);
    free(bPtr);
}