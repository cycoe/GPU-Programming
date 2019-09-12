/*  Molecular dynamics simulation linear code for binary Lennard-Jones liquid
   under NVE ensemble; Author: You-Liang Zhu, Email: youliangzhu@ciac.ac.cn
    Copyright: You-Liang Zhu
    This code is free: you can redistribute it and/or modify it under the terms
   of the GNU General Public License.*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main() {
  timeval start;              // start time
  timeval end;                // end time
  gettimeofday(&start, NULL); // get start time
  int nthreads;
  int threads_id;
/* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel num_threads(3)
  {
    /* Assign the loop into threads */
#pragma omp for
    for (int i = 0; i < 9; ++i) {
      nthreads = omp_get_num_threads();  /* Obtain the number of threads */
      threads_id = omp_get_thread_num(); /* Obtain thread number */
      printf("hello world, number of threads = %d, thread ID = %d, index= %d\n",
             nthreads, threads_id, i);
    }
  }
  gettimeofday(&end, NULL); // get end time
  long timeusr =
      (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
  printf("time is %ld  microseconds\n",
         timeusr); // the spending time on simulation in microseconds
  return 0;
}
