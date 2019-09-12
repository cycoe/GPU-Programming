/*  Molecular dynamics simulation linear code for binary Lennard-Jones liquid
   under NVE ensemble; Author: You-Liang Zhu, Email: youliangzhu@ciac.ac.cn
    Copyright: You-Liang Zhu
    This code is free: you can redistribute it and/or modify it under the terms
   of the GNU General Public License.*/
#include <ctype.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
// check CUDA error
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(-1);
  }
}

// parallel reduction
extern __shared__ int sdata[];
__global__ void compute_sums_kernel(unsigned int N, int *d_a, int *d_scratch) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int data = 0;
  if (i < N)
    data = d_a[i];

  sdata[threadIdx.x] = data;
  __syncthreads();

  int offs = blockDim.x >> 1;
  while (offs > 0) {
    if (threadIdx.x < offs)
      sdata[threadIdx.x] += sdata[threadIdx.x + offs];

    offs >>= 1;
    __syncthreads();
  }

  if (threadIdx.x == 0)
    d_scratch[blockIdx.x] = sdata[0];
}

// final parallel reduction
__global__ void compute_final_kernel(int *d_scratch,
                                     unsigned int num_partial_sums) {
  int final_sum = 0;
  for (int start = 0; start < num_partial_sums; start += blockDim.x) {
    __syncthreads();
    if (start + threadIdx.x < num_partial_sums)
      sdata[threadIdx.x] = d_scratch[start + threadIdx.x];
    else
      sdata[threadIdx.x] = 0;
    __syncthreads();

    int offs = blockDim.x >> 1;
    while (offs > 0) {
      if (threadIdx.x < offs) {
        sdata[threadIdx.x] += sdata[threadIdx.x + offs];
      }
      offs >>= 1;
      __syncthreads();
    }

    if (threadIdx.x == 0)
      final_sum += sdata[0];
  }

  if (threadIdx.x == 0)
    d_scratch[0] = final_sum;
}

int main(int argc, char **argv) {
  int N = 10000;        // the number of integers for reduction
  int *h_a;             // pointer for host memory
  int *d_a, *d_scratch; // pointer for device memory

  cudaSetDevice(0); // set GPU ID for computation

  int block_size = 64;                                    // define block size
  int n_blocks = (int)ceil((float)N / (float)block_size); // define grid size

  // Part 1 of 5: allocate host and device memory
  size_t memSize = N * sizeof(int);
  h_a = (int *)malloc(memSize);
  cudaMalloc((void **)&d_a, memSize);
  cudaMalloc((void **)&d_scratch, n_blocks * sizeof(int));

  // Part 2 of 5: initiate array
  int sum = 0;
  for (unsigned int i = 0; i < N; i++) {
    int ran = rand();
    h_a[i] = ran;
    sum += ran;
  }

  // Part 3 of 5: copy data from host to device memory
  cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy");

  // Part 4 of 5: launch kernel
  dim3 grid(n_blocks, 1, 1);
  dim3 threads(block_size, 1, 1);
  unsigned int shared_bytes = sizeof(int) * block_size;

  compute_sums_kernel<<<grid, threads, shared_bytes>>>(N, d_a, d_scratch);
  // block until the device has completed
  /* cudaThreadSynchronize(); */

  // check if kernel execution generated an error
  checkCUDAError("kernel execution 1");

  int final_block_size = 512;
  grid = dim3(1, 1, 1);
  threads = dim3(final_block_size, 1, 1);
  shared_bytes = sizeof(int) * final_block_size;

  compute_final_kernel<<<grid, threads, shared_bytes>>>(d_scratch, n_blocks);
  // block until the device has completed
  /* cudaThreadSynchronize(); */

  // check if kernel execution generated an error
  checkCUDAError("kernel execution 2");

  // Part 5 of 5: device to host copy
  cudaMemcpy(h_a, d_scratch, sizeof(int), cudaMemcpyDeviceToHost);
  // Check for any CUDA errors
  checkCUDAError("cudaMemcpy");

  // check result from device
  if (h_a[0] - sum != 0) {
    fprintf(stderr, "Failed!!! %d %d\n", h_a[0], sum);
    exit(-1);
  }

  // free device memory
  cudaFree(d_a);
  cudaFree(d_scratch);
  // free host memory
  free(h_a);
  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  printf("Success!\n");
  return 0;
}
