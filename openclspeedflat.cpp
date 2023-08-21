#include <CL/opencl.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

// seed random from c++11 random
std::random_device rd;
std::mt19937 mt(rd());
std::uniform_real_distribution<double> dist(.0001, 2.000);

/*
speed test
qty 2 prefilled array size [12k, 12k] flattened to [144M] multiplying into a
20 iterations
consistent result: 360 362 363 364 365
-
the following code is very unkempt, reader beware
*/

#define NUM_ITERATIONS 20
#define Y_SIZE 12000 // x * y must be divisible by localWorkSize, num of work groups in gpu
#define X_SIZE 12000
#define FLAT_SIZE (X_SIZE * Y_SIZE) // this is a flattened array because i couldn't quite figure out 2 dimensions

static std::string read_file( const char* fileName ) {
	std::fstream f;
	f.open( fileName, std::ios_base::in );

	std::string res;
	while( !f.eof() ) {
		char c;
		f.get( c );
		res += c;
	}
	
	f.close();

	return std::move(res);
}

cl_device_id get_gpu() {
	// get platforms
	std::cout << "\n\n";
	cl_platform_id platforms[64];
	unsigned int platformCount;
	cl_int platformResult = clGetPlatformIDs( 64, platforms, &platformCount );
	
	std::cout << "platformCount: " << platformCount << std::endl; // 1 platform
	std::cout << "success: " << platformResult << " == " << CL_SUCCESS << std::endl;
	
	
	// get devices on 0th platform
	cl_device_id devices[64];
	unsigned int deviceCount;
	cl_int deviceResult = clGetDeviceIDs( platforms[0], CL_DEVICE_TYPE_GPU, 64, devices, &deviceCount );
	
	std::cout << "deviceCount: " << deviceCount << std::endl; // 1 device
	
	
	// get device info
	char vendorName[256];
	size_t vendorNameLength;
	cl_int deviceInfoResult = clGetDeviceInfo( devices[0], CL_DEVICE_VENDOR, 256, vendorName, &vendorNameLength );
	
	std::cout << "\n";
	std::cout << "vendorName: platform[0].device[0] " << vendorName << std::endl;
	
	return devices[0];
}

int main() {
	// set device
	cl_device_id device = get_gpu();
	
	std::cout << std::setprecision(15) << "flatSize: " << FLAT_SIZE << std::endl;
	
	
	// context
	cl_int contextResult;
	cl_context context = clCreateContext( nullptr, 1, &device, nullptr, nullptr, &contextResult );
	
	cl_int commandQueueResult;
	cl_command_queue queue = clCreateCommandQueue( context, device, 0, &commandQueueResult );
	std::cout << "contextResult: " << contextResult << std::endl;
	std::cout << "commandQueueResult: " << commandQueueResult << std::endl;
	
	
	// read kernel, create program
	std::string s = read_file( "_mult_kernel.cl" );
	const char* programSource = s.c_str();
	size_t length = 0;
	cl_int programResult;
	cl_program program = clCreateProgramWithSource( context, 1, &programSource, &length, &programResult );
	std::cout << "programResult: " << programResult << std::endl;
	
	
	// program build result with debugging
	cl_int programBuildResult = clBuildProgram( program, 1, &device, "", nullptr, nullptr );
	if ( programBuildResult != CL_SUCCESS ) {
		char log[256];
		size_t logLength;
		cl_int programBuildInfoResult = clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, 256, log, &logLength );
		std::cout << log << std::endl;
	}
	//std::cout << "programBuildResult: " << programBuildResult << std::endl;
	
	
	// mem for kernel result
	cl_int kernelResult;
	cl_kernel kernel = clCreateKernel( program, "vector_mult", &kernelResult );
	std::cout << "kernelResult: " << kernelResult << std::endl;
	
	
	// malloc 2 virtual 2d arrays
	std::cout << "prefilling..." << std::endl;
	double* aPtrsData = (double*)malloc(FLAT_SIZE * sizeof(double));
	double* bPtrsData = (double*)malloc(FLAT_SIZE * sizeof(double));
	for (size_t i = 0; i < FLAT_SIZE; i++) {
		aPtrsData[i] = dist(mt);
		bPtrsData[i] = dist(mt);
	}
	std::cout << "done prefilling" << std::endl;
	/*
	for ( int i = 0; i < FLAT_SIZE; ++i ) {
		std::cout << i << "\t" << aPtrsData[i] << "\t\t" << bPtrsData[i] << std::endl;
	}
	*/
	//std::cout << aPtrsData[0] << std::endl;
	
	
	
	// for inputs I guess it is creating buffer first, then enqueue it(add to queue of work)
	cl_int aPtrsResult;
	cl_mem aPtrs = clCreateBuffer( context, CL_MEM_READ_WRITE, FLAT_SIZE * sizeof( double ), nullptr, &aPtrsResult );
	cl_int bPtrsResult;
	cl_mem bPtrs = clCreateBuffer( context, CL_MEM_READ_ONLY, FLAT_SIZE * sizeof( double ), nullptr, &bPtrsResult );
	
	// enqueue
	cl_int enqueueaPtrsResult = clEnqueueWriteBuffer( queue, aPtrs, CL_TRUE, 0, FLAT_SIZE * sizeof( double ), aPtrsData, 0, nullptr, nullptr );
	// enqueue command to write to a buffer object from host memory
	cl_int enqueuebPtrsResult = clEnqueueWriteBuffer( queue, bPtrs, CL_TRUE, 0, FLAT_SIZE * sizeof( double ), bPtrsData, 0, nullptr, nullptr );
	std::cout << "enqueueVecResult: " << enqueueaPtrsResult << " " << enqueuebPtrsResult << std::endl;

	
	
	
	// set args to kernel
	cl_int kernelArgaResult = clSetKernelArg( kernel, 0, sizeof(cl_mem), &aPtrs );
	// set arg value for specific arg of a kernel
	cl_int kernelArgbResult = clSetKernelArg( kernel, 1, sizeof(cl_mem), &bPtrs );
	std::cout << "kernelArgResult: " << kernelArgaResult << " " << kernelArgbResult << "\n\n";
	
	// timer init
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	
	// start timer
	auto t1 = high_resolution_clock::now();
	
	
	// work, local group sizes
	// this is the actual call to kernel: do work
	// the alt. here is to use clEnqueueTask which is like NDRange but with only 1 work_dim
	size_t globalWorkSize = FLAT_SIZE;
	size_t localWorkSize = 100; // use 100
	// small // 1 166ms // 8 79ms // 20 72ms // 40 69ms // 100 69ms // 200 71ms
	// large // 8 172ms // 50 154ms // 100 153ms // 200 155ms // 400 160ms
	for(size_t iter = 0; iter < NUM_ITERATIONS; iter++){
		// --------------------------------------------------------------------------------- iteration loop ---------------------------------------------------------------------------------
		// process
		cl_int enqueueKernelResult = clEnqueueNDRangeKernel( queue, kernel, 1, 0, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr );
		std::cout << "enqueueKernelResult: " << enqueueKernelResult << " -- iteration: " << iter << std::endl;
		
	}
	// read results from buffer, async style, despite that it is outside the iteration loop, the buffer from kernel automatically stores in texture
	// i.e. the results compount into aPtrs
	cl_int enqueueReadBufferResult = clEnqueueReadBuffer( queue, aPtrs, CL_FALSE, 0, FLAT_SIZE * sizeof(double), aPtrsData, 0, nullptr, nullptr );
	// from khronos docs: the 1 is a cl_uint work_dim, 0 is a const size_t *global_work_offset, last 0 is a cl_uint num_events_in_wait_list
	// work dim must be 1, 2, or 3
	
	
	// enqueue a command to read from a buffer object to host memory
	//std::cout << "enqueueReadBufferResult: " << enqueueReadBufferResult << std::endl;
	
	
	// end timer when the output has been read from buffer
	auto t2 = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(t2 - t1);
	std::cout << "\n\nTime to Process:\n" << ms_int.count() << "ms\n";
	
	// finish, whatever this does
	clFinish( queue );
	
	
	// printout results
	/*
	std::cout << "Results: \n";
	for ( int i = 0; i < FLAT_SIZE; ++i ) {
		std::cout << i << "\t" << aPtrsData[i] << "\t\t" << bPtrsData[i] << std::endl;
	}
	*/
	
	
	// release memory
	clReleaseMemObject( aPtrs );
	clReleaseMemObject( bPtrs );
	clReleaseKernel( kernel );
	clReleaseProgram( program );
	clReleaseCommandQueue( queue );
	clReleaseContext( context );
	
	return 1;
}
