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

// matrix multiplication C=A*B by device
extern "C" __global__ void matrix_kernel(float *d_c, float *d_a, float *d_b,
                                         int wa, int ha, int wb) {
  float sum = 0;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < ha && col < wb) {
    for (int i = 0; i < wa; i++) {
      sum += d_a[row * wa + i] * d_b[i * wb + col];
    }
    d_c[row * wb + col] = sum;
  }
}

// matrix multiplication C=A*B by host
void MatrixMulCPU(float *h_c, float *h_a, float *h_b, int wa, int ha, int wb) {
  float sum = 0;
  for (int i = 0; i < ha; ++i) {
    for (int j = 0; j < wb; ++j) {
      sum = 0;
      for (int k = 0; k < wa; ++k) {
        sum += (float)h_a[i * wa + k] * (float)h_b[k * wb + j];
      }
      h_c[i * wb + j] = (float)sum;
    }
  }
}

int main(int argc, char **argv) {
  int wa = 128;                 // the width of matrix A
  int ha = 64;                  // the hight of matrix A
  int wb = ha;                  // the width of matrix B
                                //	int hb = wa;
  int N = wa * ha;              // the number of elements
  float *h_a, *h_b, *h_c, *h_d; // pointer for host memory
  float *d_a, *d_b, *d_c;       // pointer for device memory

  cudaSetDevice(0); // set GPU ID for computation

  int numThreadsPerBlock = 16; // define block size
  int numBlocks_x =
      (int)ceil((float)ha / (float)numThreadsPerBlock); // define grid size
  int numBlocks_y =
      (int)ceil((float)wb / (float)numThreadsPerBlock); // define grid size

  // Part 1 of 5: allocate host and device memory
  size_t memSize = N * sizeof(float);

  h_a = (float *)malloc(memSize);
  h_b = (float *)malloc(memSize);
  h_c = (float *)malloc(ha * wb * sizeof(float));
  h_d = (float *)malloc(ha * wb * sizeof(float));
  cudaMalloc((void **)&d_a, memSize);
  cudaMalloc((void **)&d_b, memSize);
  cudaMalloc((void **)&d_c, ha * wb * sizeof(float));

  // Part 2 of 5: initiate host array
  for (unsigned int i = 0; i < N; i++) {
    int ran = rand();
    float fran = (float)ran / (float)RAND_MAX;
    h_a[i] = fran;

    ran = rand();
    fran = (float)ran / (float)RAND_MAX;
    h_b[i] = fran;
  }

  //  matrix multiplication by host
  MatrixMulCPU(h_d, h_a, h_b, wa, ha, wb);

  // Part 3 of 5: copy data from host to device
  cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, memSize, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy");

  // Part 4 of 5: launch kernel
  dim3 dimGrid(numBlocks_x, numBlocks_y);
  dim3 dimBlock(numThreadsPerBlock, numThreadsPerBlock);

  matrix_kernel<<<dimGrid, dimBlock>>>(d_c, d_a, d_b, wa, ha, wb);

  cudaThreadSynchronize(); // block until the device has completed

  // check if kernel execution generated an error
  checkCUDAError("kernel execution");

  // Part 5 of 5: device to host copy
  cudaMemcpy(h_c, d_c, ha * wb * sizeof(float), cudaMemcpyDeviceToHost);
  // Check for any CUDA errors
  checkCUDAError("cudaMemcpy");

  // check the results from device
  for (unsigned int i = 0; i < ha * wb; i++) {
    if (abs(h_c[i] - h_d[i]) > 0.00001) {
      fprintf(stderr, "Failed!!! %d %f %f\n", i, h_c[i], h_d[i]);
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
  free(h_d);
  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  printf("Success!\n");
  return 0;
}
