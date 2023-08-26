#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>


/*
speed test
this is version 2 of the single thread c, which is at least 5 secs faster than the flat array
qty 2 prefilled array size [12k, 12k] multiplying into the first,
20 iterations
consistent result: 2040 2150 2180 2180 2410 2440 2590
*/

#define SIZE_Y_ARRAY 12000
#define SIZE_X_ARRAY 12000
#define NUM_ITERATIONS 20

struct funcArgs {
    double** aArrYPtrs;
    double** bArrYPtrs;
};


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

void* process(void* input){
    for(int y = 0; y < SIZE_Y_ARRAY; y++){
        for(int x = 0; x < SIZE_X_ARRAY; x++){
            // building on speedtest2.c, we have to pass in a struct since pthread only accepts a single void* arg
            ((struct funcArgs*)input)->aArrYPtrs[y][x] *= ((struct funcArgs*)input)->bArrYPtrs[y][x];
        }
    }
}

int main(){
    // seed random
    time_t t;
    srand((unsigned) time(&t));
    
    // first malloc a pointer to a struct funcArgs, then continue on with whatever was in speedtest2.c
    // malloc a pointer to a pointer, the first array is the y direction
    struct funcArgs* TheseArgs = (struct funcArgs*)malloc(sizeof(struct funcArgs));
    TheseArgs->aArrYPtrs = malloc(SIZE_Y_ARRAY * sizeof(double));
    TheseArgs->bArrYPtrs = malloc(SIZE_Y_ARRAY * sizeof(double));
    
    // prefill 2 array with random values
    prefill(TheseArgs->aArrYPtrs);
    prefill(TheseArgs->bArrYPtrs);
    printf("done prefilling\n");
    
    //debugLoopThru(TheseArgs->aArrYPtrs);
    //debugLoopThru(TheseArgs->bArrYPtrs);
    
    // malloc pthreads
    pthread_t* threadIdPtrs = malloc(NUM_ITERATIONS * sizeof(pthread_t));

    // start timer and loop
    clock_t timer;
    timer = clock();
    for(int i = 0; i < NUM_ITERATIONS; i++){
        pthread_create(threadIdPtrs + i, NULL, process, (void*)TheseArgs);
        //process(aArrYPtrs, bArrYPtrs);
    }
    
    // wait for all the threads to sync
    for(int i = 0; i < NUM_ITERATIONS; i++){
        pthread_join(*(threadIdPtrs + i), NULL);
    }
    timer = clock() - timer;
    

    // get processor time, otherwise it returns 0 sec
    double timeTaken = ((double)timer) / CLOCKS_PER_SEC;
    

    // debug test if it actually did it
    //debugLoopThru(TheseArgs->aArrYPtrs);

    printf("%.3f secs\n", timeTaken);
    
    free(TheseArgs);
    pthread_exit(NULL);
    return 0;
}