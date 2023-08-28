import cupy as gp
import numpy as np
import time

# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# 20 iterations: system is cheating, as if I printout the value of a single cell in a_arr, it *conveniently* takes 4x longer
# using the "realistic" values: 270 270 270 280 280, these still seem a bit cheaty, and I have no confidence it is reporting an accurate number, so no score

Y_SIZE = 12000
X_SIZE = 12000
ITERATIONS = 20


def process(a, b):
    return gp.multiply(a, b)


process_kernel = gp.RawKernel(r'''
extern "C" __global__
void my_mult(const double* a, const double* b, double* c) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    c[tid] = a[tid] * b[tid];
}
''', 'my_mult') # this does work if you call process_kernel((y,), (y,), (a_arr, b_arr, a_arr)) where y is 1, 500, 1000
# but if you are going to write C kernels, just switch to C++ fam


element_process_kernel = gp.ElementwiseKernel(
    'float64 a, float64 b',
    'float64 c',
    'c = a * b',
    'element_process_kernel'
)


# create 2d array from random
a_arr = gp.random.uniform(0, 2, (Y_SIZE, X_SIZE), dtype=np.float64)
b_arr = gp.random.uniform(0, 2, (Y_SIZE, X_SIZE), dtype=np.float64)
print("done prefilling")

# start timer and exec
start = time.perf_counter()
for iter in range(ITERATIONS):
    #print(iter)
    #a_arr = np.multiply(a_arr, b_arr)
    element_process_kernel(a_arr, b_arr, a_arr)
    print("a ", a_arr[0][0]) # for some reason including this line here makes the results more "realistic", like it cheats if you don't witness some results
    
# stop timer
end = time.perf_counter()
print("{:.3f}".format(end - start), "secs")

# kernel is for sure cheating, so make it do extra work after the test
a_arr = element_process_kernel(a_arr, b_arr)