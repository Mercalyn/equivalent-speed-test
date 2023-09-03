import cupy as gp
import numpy as np
import time
import math

# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# 20 iterations: 200 200 200 200 200

Y_SIZE = 12000
X_SIZE = 12000
ITERATIONS = 20


def process(a, b):
    return gp.multiply(a, b) # this works, but wanted to see other ways to write kernels


process_kernel = gp.RawKernel(r'''
extern "C" __global__
void my_mult(double* a, const double* b, int xWidth, int total) {
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;
    int gid = (y * xWidth) + x;
                              
    if(gid < total){
        a[gid] *= b[gid];
    }
}
''', 'my_mult') # if you are going to write C kernels, just switch to C++ fam


element_process_kernel = gp.ElementwiseKernel(
    'raw float64 a, float64 b', # by adding raw in front of a, you can then manually index
    'float64 c',
    'c = a[7]', # flat array style, despite it being a 2d array, in kernel it is flat
    'element_process_kernel'
) # this has less control over a lot of dimensions


# create 2d array from random
print("prefilling... ")
a_arr = gp.random.uniform(0, 2, (Y_SIZE, X_SIZE), dtype=np.float64)
b_arr = gp.random.uniform(0, 2, (Y_SIZE, X_SIZE), dtype=np.float64)
print(a_arr[Y_SIZE - 1][X_SIZE - 1])
print(b_arr[Y_SIZE - 1][X_SIZE - 1])
#print(b_arr)

# dims and sizes
numThreads = (32, 32) # threads per block, 2d max 32, 32
numBlocks = ( # blocks per grid
    math.ceil(Y_SIZE / numThreads[0]), 
    math.ceil(X_SIZE / numThreads[1]),
)

# start timer and exec
print("processing... ")
start = time.perf_counter()
for iter in range(ITERATIONS):
    #print(iter)
    process_kernel(numBlocks, numThreads, (a_arr, b_arr, X_SIZE, (X_SIZE * Y_SIZE))) #(y,), (x,)
    #element_process_kernel(a_arr, b_arr, a_arr)
    
gp.cuda.runtime.deviceSynchronize()

# stop timer
end = time.perf_counter()
print("{:.2f}".format((end - start) * 1000), "ms")
print()

#print(a_arr)
print(a_arr[Y_SIZE - 1][X_SIZE - 1])