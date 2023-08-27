import numpy as np
from numba import jit
import time

# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# 20 iterations: 6180 6250 6260 6330 6550
# I am not sure why these results are worse than just np, maybe it is the
# extra numba overhead

Y_SIZE = 12000
X_SIZE = 12000
ITERATIONS = 20


@jit(nopython=True)
def process(a, b):
    # numba likes loops, np maths
    return a * b # this is called broadcasting in numpy, numba likes this
    #return np.multiply(a, b) # this has the same performance as above


# create 2d array from random float64
a_arr = np.random.uniform(0, 2, (Y_SIZE, X_SIZE))
b_arr = np.random.uniform(0, 2, (Y_SIZE, X_SIZE))
print("done prefilling... ")
#print(a_arr)
#print(b_arr)


# numba needs to cache precompilation by calling things first before timer:
process(a_arr, b_arr) # without precompilation times: 6620ms, about 350ms slower
print("done precompilation... ")


# start timer and exec
start = time.perf_counter()
for iter in range(ITERATIONS):
    a_arr = process(a_arr, b_arr)
    
    
# stop timer
end = time.perf_counter()
print("{:.3f}".format(end - start), "secs")

# debug output to show work done
#print(a_arr)