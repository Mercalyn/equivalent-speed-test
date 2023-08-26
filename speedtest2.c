#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


/*
speed test
this is version 2 of the single thread c, which is at least 5 secs faster than the flat array
qty 2 prefilled array size [12k, 12k] multiplying into the first,
20 iterations
consistent result: 10290 10390 10500 10820 10900
*/

#define SIZE_Y_ARRAY 12000
#define SIZE_X_ARRAY 12000
#define NUM_ITERATIONS 20


void prefill(double** arrayPtr){
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        /*
        malloc another array inside of each item in the y, on the x direction
        notice that there is only 1 * deref here, that is because we are still setting
        the address of something(malloc returns address)
        */
        *(arrayPtr + y) = malloc(SIZE_X_ARRAY * sizeof(double));
        
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            /*
            so what we are given is a pointer to a pointer of some double value.
            the base pointer, arrayPtr is an address that points to the y axis, and is thus
            offset by the y, then it resolves once into a regular, single pointer that points to
            an address on the x axis, so that is offset by its x, which is finally
            resolved into assigning something into the correct address
            */
            *( *(arrayPtr + y) + x) = (( (double)rand() / (double)(RAND_MAX) ) * 2) + .00001;
        }
    }
}

void debugLoopThru(double** arrayPtr){
    printf("\n\n");
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        //printf("y: %d\n", y);
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            //printf("x: %d\n", x);
            //printf("%p\n", ( *(arrayPtr + y) + x));
            printf("%f\n", *( *(arrayPtr + y) + x));
        }
    }
}

void process(double** aArrYPtrs, double** bArrYPtrs){
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            // what a lovely mess of stars, see prefill to understand it
            //*( *(aArrYPtrs + y) + x) *= *( *(bArrYPtrs + y) + x);
            
            // this essentially does the same thing as above with no speed penalty
            aArrYPtrs[y][x] *= bArrYPtrs[y][x];
        }
    }
}

int main(){
    // seed random
    time_t t;
    srand((unsigned) time(&t));
    
    // malloc a pointer to a pointer, the first array is the y direction
    double** aArrYPtrs = malloc(SIZE_Y_ARRAY * sizeof(double));
    double** bArrYPtrs = malloc(SIZE_Y_ARRAY * sizeof(double));
    
    // prefill 2 array with random values
    prefill(aArrYPtrs);
    prefill(bArrYPtrs);
    printf("done prefilling\n");
    
    //debugLoopThru(aArrYPtrs);
    //debugLoopThru(bArrYPtrs);

    // start timer and loop
    clock_t timer;
    timer = clock();
    for(int i = 0; i < NUM_ITERATIONS; i++){
        process(aArrYPtrs, bArrYPtrs);
    }
    timer = clock() - timer;

    // get processor time, otherwise it returns 0 sec
    double timeTaken = ((double)timer) / CLOCKS_PER_SEC;
    

    // debug test if it actually did it
    //debugLoopThru(aArrYPtrs);

    printf("%.3f secs\n", timeTaken);
    
    free(aArrYPtrs);
    free(bArrYPtrs);
}