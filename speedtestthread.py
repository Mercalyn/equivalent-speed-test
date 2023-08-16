import numpy as np
import time
from multiprocessing.pool import ThreadPool as Pool

# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# multiproc version
# 20 iterations: 4120 4130 4280 4290 4300 4320 4520

Y_SIZE = 12000
X_SIZE = 12000
ITERATIONS = 20 # also number of pools in multiproc
pool = Pool(ITERATIONS)


def process(a_arr, b_arr):
    a_arr = np.multiply(a_arr, b_arr)
    return a_arr


if __name__ == "__main__":
    # create 2d array from random, it says it is a float64, and yet, it seems to capout at 8 decimals?
    a_arr = np.random.uniform(0, 2, (Y_SIZE, X_SIZE))
    b_arr = np.random.uniform(0, 2, (Y_SIZE, X_SIZE))
    #print(a_arr)
    #print(b_arr)
    print("done prefilling")
    # no actually it is float64, but it caps the values only when it prints it out
    #for y in range(Y_SIZE):
        #for x in range(X_SIZE):
            #print("hh", a_arr[y, x])


    # start timer and exec
    start = time.perf_counter()
    for iter in range(ITERATIONS):
        #a_arr = process(a_arr.view(), b_arr.view())
        # start async pools
        a_arr = pool.apply_async(process, (a_arr, b_arr))
    
    # waiting for all pools to finish
    pool.close()
    pool.join()
        
        
    # stop timer
    end = time.perf_counter()
    print("{:.3f}".format(end - start), "secs")
        
    #print(a_arr)