import numpy as np
import pyopencl as cl
import time

# speed test
# qty 2 prefilled array size [12k, 12k] multiplying into the first,
# 20 iterations: 840 840 850 860 870 880 910

Y_SIZE = 12000
X_SIZE = 12000
ITERATIONS = 20

# opencl setup
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=[my_gpu_devices[0]])
queue = cl.CommandQueue(ctx)

# data prefill
a_arr = np.random.uniform(0, 2, (Y_SIZE, X_SIZE))
b_arr = np.random.uniform(0, 2, (Y_SIZE, X_SIZE))
print("done prefilling")
#c_arr = np.empty_like(a_arr) # not using dedicated out only
#print(a_arr)
#print(b_arr)

# i/o buffers, cl buffers are kind of like gpu.js textures
mf = cl.mem_flags
a_buffer = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_arr)
b_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_arr)

# program.kernel
cl_programs = cl.Program(ctx, """
__kernel void mult(__global double* a, __global const double* b) {
    // flat index
    int x = (get_global_id(1) * get_global_size(0)) + get_global_id(0);
    a[x] *= b[x];
}
""").build()

# output buffer
#c_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, a_arr.nbytes)

# kernel call and start timer
start = time.perf_counter()
for iter in range(ITERATIONS):
    cl_programs.mult(queue, a_arr.shape, None, a_buffer, b_buffer)
    #a_buffer = c_buffer # set a to prior c results

# send back to host mem
cl.enqueue_copy(queue, a_arr, a_buffer)

# stop timer
end = time.perf_counter()
print("{:.2f}".format((end - start) * 1000), "ms")

#print(a_arr)
