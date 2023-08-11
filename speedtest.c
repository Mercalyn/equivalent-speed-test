#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
see goldrush folder for cpu vs gpu speed test 2
100k iterations of 1 dimensional array of 12k all doing 
(1.78939423494 * 0.785186454) + (4.113810156 * .500005123)
*/

int const SIZE_X_ARRAY = 16000;
int const NUM_ITERATIONS = 100000;

void process(int* startArrayPtr){
    for(int x = 0; x < SIZE_X_ARRAY; x++){
        // each item in x array, do the following process
        *(startArrayPtr + x) = (1.78939423494 * 0.785186454) + (4.113810156 * .500005123);
    }
}

int main(){
    // malloc
    int* startArrayPtr = malloc(SIZE_X_ARRAY * sizeof(int));


    // start timer and loop
    clock_t timer;
    timer = clock();
    for(int i = 0; i < NUM_ITERATIONS; i++){
        process(startArrayPtr);
    }
    timer = clock() - timer;

    // get processor time, otherwise it returns 0 sec
    double timeTaken = ((double)timer) / CLOCKS_PER_SEC;

    /*
    // debug test if it actually did it
    for(int i = 0; i < SIZE_X_ARRAY; i++){
        printf("%d\n", *(startArrayPtr + i));
    }
    */

    printf("%.3f secs\n", timeTaken);
}