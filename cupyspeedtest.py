import cupy as gp
import numpy as np
import time

# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# 20 iterations: nothing outside of the 0th grid/thread will actually calc...
# see http://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
# but something in CuPy is limiting me from allocating anywhere lcose to the maximum number of block*threads that I need
# I seem to be limited to 20 blocks of 20 threads in x, and y, which is not a lot of data....

Y_SIZE = 3
X_SIZE = 6
ITERATIONS = 1


def process(a, b):
    return gp.multiply(a, b) # this works, but wanted to see other ways to write kernels


process_kernel = gp.RawKernel(r'''
extern "C" __global__
void my_mult(const double* a, const double* b, double* c, int xWidth) {
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;
    int f = (y * xWidth) + x;
    //c[f] = a[f] * b[f];
    c[f] = f;
}
''', 'my_mult') # if you are going to write C kernels, just switch to C++ fam


element_process_kernel = gp.ElementwiseKernel(
    'float64 a, float64 b',
    'float64 c',
    'c = a * b',
    'element_process_kernel'
) # this has less control over a lot of dimensions


# create 2d array from random
a_arr = gp.random.uniform(0, 2, (Y_SIZE, X_SIZE), dtype=np.float64)
b_arr = gp.random.uniform(0, 2, (Y_SIZE, X_SIZE), dtype=np.float64)
c_arr = a_arr
print(a_arr)
print(b_arr)
print("done prefilling")

#
tpb = (2, 2) # threads per block
bpg = (2, 2) # blocks per grid, these are set to max number I can have them at

# start timer and exec
start = time.perf_counter()
for iter in range(ITERATIONS):
    #print(iter)
    process_kernel(bpg, tpb, (a_arr, b_arr, c_arr, X_SIZE)) #(y,), (x,)
    #element_process_kernel(a_arr, b_arr, a_arr)
    a_arr = c_arr
    print(c_arr)
    
# stop timer
end = time.perf_counter()
print("{:.2f}".format((end - start) * 1000), "ms")

print(c_arr)