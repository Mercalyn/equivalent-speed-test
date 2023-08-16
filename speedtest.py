import numpy as np
import time

# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# 20 iterations: 5890 5910 5930 5980 5980 6000 6510

Y_SIZE = 12000
X_SIZE = 12000
ITERATIONS = 20

# create 2d array from random, it says it is a float64, and yet, it seems to capout at 8 decimals?
a_arr = np.random.uniform(0, 2, (Y_SIZE, X_SIZE))
b_arr = np.random.uniform(0, 2, (Y_SIZE, X_SIZE))
print("done prefilling")
# no actually it is float64, but it caps the values only when it prints it out
#for y in range(Y_SIZE):
    #for x in range(X_SIZE):
        #print("hh", a_arr[y, x])


# start timer and exec
start = time.perf_counter()
for iter in range(ITERATIONS):
    # inside num iterations
    #print(iter)
    a_arr = np.multiply(a_arr, b_arr)
    
# stop timer
end = time.perf_counter()
print("{:.3f}".format(end - start), "secs")
    
#print(a_arr)