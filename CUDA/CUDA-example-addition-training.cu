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

// addition C = A + B
__global__ void myFirstKernel(int N, int *d_a, int *d_b, int *d_c) {
  // write codes here
    for (int i = 0; i < N; i++) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main(int argc, char **argv) {
  int *h_a, *h_b, *h_c; // pointer for host memory
  int *d_a, *d_b, *d_c; // pointer for device memory
  int N = 10000;        // number of integers

  cudaSetDevice(0); // set GPU ID for computation

  // threads per block is 2^n
  int numThreadsPerBlock = 128; // define block size
  int numBlocks =
      (int)ceil((float)N / (float)numThreadsPerBlock); // define grid size

  // Part 1 of 5: allocate host and device memory
  size_t memSize = N * sizeof(int);

  h_a = (int *)malloc(memSize);
  h_b = (int *)malloc(memSize);
  h_c = (int *)malloc(memSize);
  cudaMalloc((void **)&d_a, memSize);
  cudaMalloc((void **)&d_b, memSize);
  cudaMalloc((void **)&d_c, memSize);

  // Part 2 of 5: initiate data
  for (unsigned int i = 0; i < N; i++) {
    h_a[i] = i;
    h_b[i] = N - i;
  }

  // Part 3 of 5: copy data from host to device
  cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, memSize, cudaMemcpyHostToDevice);

  // Part 4 of 5: launch kernel
  dim3 dimGrid(numBlocks);
  dim3 dimBlock(numThreadsPerBlock);
  myFirstKernel<<<dimGrid, dimBlock>>>(N, d_a, d_b, d_c);

  cudaThreadSynchronize(); // block until the device has completed

  // check if kernel execution generated an error
  checkCUDAError("kernel execution");

  // Part 5 of 5: device to host copy
  cudaMemcpy(h_c, d_c, memSize, cudaMemcpyDeviceToHost);
  // Check for any CUDA errors
  checkCUDAError("cudaMemcpy");

  // check results from device
  for (unsigned int i = 0; i < N; i++) {
    if (h_c[i] != N) {
      fprintf(stderr, "Failed!!! %d %d %d\n", i, h_c[i], N);
      exit(-1);
    }
  }
  // free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  // free host memory
  free(h_a);
  free(h_b);
  free(h_c);
  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  printf("Success!\n");
  return 0;
}
