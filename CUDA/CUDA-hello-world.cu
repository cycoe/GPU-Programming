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
// periodic boundary condition

__global__ void myFirstKernel() {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 9)
    printf("hello world, thread %d \n", idx);
}

int main() {
  timeval start;              // start time
  timeval end;                // end time
  gettimeofday(&start, NULL); // get start time
  dim3 dimGrid(1);            // grid
  dim3 dimBlock(32);          // block

  // kernel function
  myFirstKernel<<<dimGrid, dimBlock>>>();

  // block until the device has completed
  cudaThreadSynchronize();
  gettimeofday(&end, NULL); // get end time
  long timeusr =
      (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
  printf("time is %ld  microseconds\n",
         timeusr); // the spending time on simulation in microseconds
  return 0;
}
