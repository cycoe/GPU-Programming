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

// catch the error thrown by CUDA
void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(-1);
  }
}

// implement periodic boundary condition
__host__ __device__ float pbc(float x, float box_len) {
  float box_half = box_len * 0.5;
  if (x > box_half)
    x -= box_len;
  else if (x < -box_half)
    x += box_len;
  return x;
}

// random number generator [0.0-1.0)
float R2S() {
  int ran = rand();
  float fran = (float)ran / (float)RAND_MAX;
  return fran;
}

// initially generate the position and mass of particles
void init(unsigned int np, float4 *r, float4 *v, float3 box, float min_dis) {
  for (unsigned int i = 0; i < np; i++) {
    bool find_pos = false;
    float4 ri;
    while (!find_pos) {
      ri.x = (R2S() - 0.5) * box.x;
      ri.y = (R2S() - 0.5) * box.y;
      ri.z = (R2S() - 0.5) * box.z;
      find_pos = true;

      for (unsigned int j = 0; j < i; j++) {
        float dx = pbc(ri.x - r[j].x, box.x);
        float dy = pbc(ri.y - r[j].y, box.y);
        float dz = pbc(ri.z - r[j].z, box.z);
        float r = sqrt(dx * dx + dy * dy + dz * dz);
        // a minimum safe distance to avoid the overlap of LJ particles
        if (r < min_dis) {
          find_pos = false;
          break;
        }
      }
    }
    // randomly generate the type of particle, 1.0 represent type A and 2.0
    // represent type B
    ri.w = R2S() > 0.5 ? 1.0 : 2.0;

    r[i] = ri;
    v[i].w = 1.0;
  }
}

// first step integration of velocity verlet algorithm
extern "C" __global__ void first_integration_kernel(unsigned int np, float dt,
                                                    float3 box, float4 *r,
                                                    float4 *v, float4 *f) {
  // calculate the global index of thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < np) {
    float4 ri = r[i];
    float mass = v[i].w;

    v[i].x += 0.5 * dt * f[i].x / mass;
    v[i].y += 0.5 * dt * f[i].y / mass;
    v[i].z += 0.5 * dt * f[i].z / mass;

    ri.x += dt * v[i].x;
    ri.y += dt * v[i].y;
    ri.z += dt * v[i].z;

    r[i].x = pbc(ri.x, box.x);
    r[i].y = pbc(ri.y, box.y);
    r[i].z = pbc(ri.z, box.z);
  }
}

void first_integration(unsigned int np, float dt, float3 box, float4 *r,
                       float4 *v, float4 *f, unsigned int nthreads) {
  dim3 grid((np / nthreads) + 1, 1, 1);
  dim3 block(nthreads, 1, 1);

  first_integration_kernel<<<grid, block>>>(np, dt, box, r, v, f);
  // block until the device has complete
  cudaDeviceSynchronize();

  // check if kernel function execution throw out an error
  checkCUDAError("Kernel execution");
}

// non-bonded force calculation
extern "C" __global__ void force_calculation_kernel(unsigned int np, float3 box,
                                                    float3 lj1, float3 lj2,
                                                    float4 *r, float4 *f,
                                                    float rcutsq) {
  // declare an shared array
  extern __shared__ float4 spos[];
  // declare i for global thread index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float4 force = make_float4(0.0, 0.0, 0.0, 0.0);
  float4 ri = make_float4(0.0, 0.0, 0.0, 0.0);

  // if index is less than the index of particles
  if (i < np)
    ri = r[i];

  // copy data from global memory to shared memory by one block of data at a
  // time
  for (int start = 0; start < np; start += blockDim.x) {
    float4 pos = make_float4(0.0, 0.0, 0.0, 0.0);
    // the thread with ID x will copy the data with global ID
    // (threadIdx.x + start)th to its block's shared memory
    if (start + threadIdx.x < np)
      pos = r[start + threadIdx.x];
    __syncthreads();
    spos[threadIdx.x] = pos;
    __syncthreads();

    // end_offset is the biggest offset of last block
    int end_offset = min(blockDim.x, np - start);

    if (i < np) {
      for (unsigned int offset = 0; offset < end_offset; offset++) {
        int j = start + offset;

        /* particles have no interactions with themselves */
        if (i == j)
          continue;

        float4 rj = spos[offset];

        /* calculated the shortest distance between particle i and j */
        float dx = pbc(ri.x - rj.x, box.x);
        float dy = pbc(ri.y - rj.y, box.y);
        float dz = pbc(ri.z - rj.z, box.z);
        float type = ri.w + rj.w;

        float rsq = dx * dx + dy * dy + dz * dz;

        /* compute force and energy if within cutoff */
        if (rsq < rcutsq) {
          float lj1_ij, lj2_ij;
          if (type == 2.0) // i=1.0, j=1.0
          {
            lj1_ij = lj1.x;
            lj2_ij = lj2.x;
          } else if (type == 3.0) // i=1.0, j=2.0; or i=2.0, j=1.0
          {
            lj1_ij = lj1.y;
            lj2_ij = lj2.y;
          } else if (type == 4.0) // i=2.0, j=2.0
          {
            lj1_ij = lj1.z;
            lj2_ij = lj2.z;
          }

          // force transform to float is necessary here
          // float calculation is much faster than double
          float r2inv = float(1.0) / rsq;
          float r6inv = r2inv * r2inv * r2inv;

          float ffac = r2inv * r6inv *
                       (float(12.0) * lj1_ij * r6inv - float(6.0) * lj2_ij);
          float epot = r6inv * (lj1_ij * r6inv - lj2_ij);

          force.x += ffac * dx;
          force.y += ffac * dy;
          force.z += ffac * dz;
          force.w += epot;
        }
      }
    }
  }
  if (i < np)
    f[i] = force;
}

void force_calculation(unsigned int np, float3 box, float3 lj1, float3 lj2,
                       float4 *r, float4 *f, float rcutsq,
                       unsigned int nthreads) {
  dim3 grid((np / nthreads) + 1, 1, 1);
  dim3 block(nthreads, 1, 1);

  unsigned int shared_bytes = nthreads * sizeof(float4);
  force_calculation_kernel<<<grid, block, shared_bytes>>>(np, box, lj1, lj2, r,
                                                          f, rcutsq);
  cudaDeviceSynchronize();
  checkCUDAError("Kernel execution");
}

// second step integration of velocity verlet algorithm
__global__ void second_integration_kernel(unsigned int np, float dt, float4 *v,
                                          float4 *f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < np) {
    float mass = v[i].w;
    v[i].x += 0.5 * dt * f[i].x / mass;
    v[i].y += 0.5 * dt * f[i].y / mass;
    v[i].z += 0.5 * dt * f[i].z / mass;
  }
}

void second_integration(unsigned int np, float dt, float4 *v, float4 *f,
                        unsigned int nthreads) {
  dim3 grid((np / nthreads) + 1, 1, 1);
  dim3 block(nthreads, 1, 1);

  second_integration_kernel<<<grid, block>>>(np, dt, v, f);

  cudaDeviceSynchronize();
  checkCUDAError("Kernel execution");
}

// system information collection for temperature, kinetic energy, potential and
// total energy
__global__ void compute_info_threads(unsigned int np, float4 *v, float4 *f,
                                     float2 *scratch) {
  extern __shared__ float2 sdata[];
  int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

  float2 temp = make_float2(0.0, 0.0);
  if (i < np) {
    float4 vi = v[i];
    temp.x = vi.w * (vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
    temp.y = f[i].w;
  }

  if (i + blockDim.x < np) {
    float4 vi = v[i + blockDim.x];
    temp.x += vi.w * (vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
    temp.y += f[i + blockDim.x].w;
  }

  sdata[threadIdx.x] = temp;
  __syncthreads();

  // divide and rule
  int offset = blockDim.x >> 1;
  while (offset > 0) {
    if (threadIdx.x < offset) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + offset].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + offset].y;
    }
    offset >>= 1;
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scratch[blockIdx.x].x = sdata[0].x;
    scratch[blockIdx.x].y = sdata[0].y;
  }
}

__global__ void compute_info_blocks(unsigned int np, float *info,
                                    float2 *scratch, unsigned int nblocks) {
  extern __shared__ float2 sdata[];
  float2 final_sum = make_float2(0.0, 0.0);

  for (int start = 0; start < nblocks; start += blockDim.x * 2) {
    float2 temp = make_float2(0.0, 0.0);
    if (start + threadIdx.x < nblocks) {
      temp.x = scratch[start + threadIdx.x].x;
      temp.y = scratch[start + threadIdx.x].y;
      if (start + threadIdx.x + blockDim.x < nblocks) {
        temp.x = scratch[start + threadIdx.x + blockDim.x].x;
        temp.y = scratch[start + threadIdx.x + blockDim.x].y;
      }
    }

    sdata[threadIdx.x] = temp;
    __syncthreads();

    int offset = blockDim.x >> 1;
    while (offset > 0) {
      if (threadIdx.x < offset) {
        sdata[threadIdx.x].x += sdata[threadIdx.x + offset].x;
        sdata[threadIdx.x].y += sdata[threadIdx.x + offset].y;
      }
      offset >>= 1;
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      final_sum.x += sdata[0].x;
      final_sum.y += sdata[0].y;
    }
  }

  if (threadIdx.x == 0) {
    float ekin = 0.5 * final_sum.x;
    float potential = 0.5 * final_sum.y;
    unsigned int nfreedom = 3 * np - 3;
    float temper = 2.0 * ekin / float(nfreedom);
    float energy = ekin + potential;

    info[0] = temper;
    info[1] = potential;
    info[2] = energy;
  }
}

void compute_info(unsigned int np, float4 *v, float4 *f, float *info,
                  float2 *scratch, unsigned int nthreads,
                  unsigned int nblocks) {
  dim3 grid(nblocks, 1, 1);
  dim3 block(nthreads >> 1, 1, 1);
  unsigned int shared_bytes = sizeof(float2) * nthreads >> 1;

  compute_info_threads<<<grid, block, shared_bytes>>>(np, v, f, scratch);
  cudaDeviceSynchronize();
  checkCUDAError("kernel execution");

  int offset = 0;
  int temp = nblocks;
  while (temp > 0) {
    temp >>= 1;
    offset += 1;
  }
  unsigned int final_nthreads = 512;
  shared_bytes = sizeof(float2) * final_nthreads;
  grid = dim3(1, 1, 1);
  block = dim3(final_nthreads, 1, 1);

  compute_info_blocks<<<grid, block, shared_bytes>>>(np, info, scratch,
                                                     nblocks);
  cudaDeviceSynchronize();
  checkCUDAError("kernel execution");
}

// output system information and frame in XYZ formation which can be read by VMD
void output(FILE *traj, unsigned int step, float *info, float4 *r,
            unsigned int np) {
  float temp = info[0];
  float potential = info[1];
  float energy = info[2];

  fprintf(traj, "%d\n  step=%d  temp=%20.8f  pot=%20.8f  ener=%20.8f\n", np,
          step, temp, potential, energy);
  for (unsigned int i = 0; i < np; i++) {
    float4 ri = r[i];
    if (ri.w == 1.0)
      fprintf(traj, "A  %20.8f %20.8f %20.8f\n", ri.x, ri.y, ri.z);
    else if (ri.w == 2.0)
      fprintf(traj, "B  %20.8f %20.8f %20.8f\n", ri.x, ri.y, ri.z);
  }
}

// main function
int main(int argc, char **argv) {
  // running parameters
  unsigned int np = 2700;    // the number of particles
  unsigned int nsteps = 500; // the number of time steps
  float dt = 0.001;          // integration time step
  float rcut = 3.0;          // the cutoff radius of interactions
  // float temperature = 1.0;// target temperature
  unsigned int nprint = 100; //  period for data output

  timeval start; // start time
  timeval end;   // end time

  // box size in x, y, and z directions
  float3 box = make_float3(15.0, 15.0, 15.0);
  // epsilon.x for type 1.0 and 1.0; epsilon.y for
  // type 1.0 and 2.0; epsilon.z for type 1.0 and 2.0
  float3 epsilon = make_float3(1.0, 0.5, 1.0);
  // sigma.x for type 1.0 and 1.0; sigma.y for
  // type 1.0 and 2.0; sigma.z for type 1.0 and 2.0
  float3 sigma = make_float3(1.0, 1.0, 1.0);
  // the minimum distance between particles for system generation
  float min_dis = sigma.x * 0.9;

  float3 lj1, lj2;

  // calculate these constants firstly
  lj1.x = 4.0 * epsilon.x * pow(sigma.x, int(12));
  lj1.y = 4.0 * epsilon.y * pow(sigma.y, int(12));
  lj1.z = 4.0 * epsilon.z * pow(sigma.z, int(12));

  lj2.x = 4.0 * epsilon.x * pow(sigma.x, int(6));
  lj2.y = 4.0 * epsilon.y * pow(sigma.y, int(6));
  lj2.z = 4.0 * epsilon.z * pow(sigma.z, int(6));

  // announce GPU device ID
  cudaSetDevice(0);

  // number of threads per block
  unsigned int nthreads = 64;
  // number of blocks
  unsigned int nblocks = (int)ceil((float)np / (float)nthreads);
  // memory size
  size_t memSize = np * sizeof(float4);

  // memory allocation
  float4 *r = (float4 *)malloc(memSize); // rx, ry, rz, type(0, 1, 2 ...)
  float4 *v = (float4 *)malloc(memSize); // vx, vy, vz, mass
  float4 *f = (float4 *)malloc(memSize); // fx, fy, fz, potential
  float *info =
      (float *)malloc(16 * sizeof(float)); // temperature, potential, energy ...

  float4 *r_d = NULL;
  float4 *v_d = NULL;
  float4 *f_d = NULL;
  float *info_d = NULL;
  float2 *scratch_d = NULL;
  // memory allocation in GPU memory
  cudaMalloc((void **)&r_d, memSize);
  cudaMalloc((void **)&v_d, memSize);
  cudaMalloc((void **)&f_d, memSize);
  cudaMalloc((void **)&info_d, 16 * sizeof(float));
  cudaMalloc((void **)&scratch_d, nblocks * sizeof(float));

  // trajectory file in XYZ format that can be open by VMD
  FILE *traj = fopen("traj.cu.xyz", "w");

  /* generate system information */
  printf("Starting simulation with %d atoms for %d steps.\n", np, nsteps);
  printf("Generating system.\n");

  // initiate some particles in box
  init(np, r, v, box, min_dis);

  // copy location and velocity into GPU memory
  cudaMemcpy(r_d, r, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(v_d, v, memSize, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy");

  // get start time
  gettimeofday(&start, NULL);

  /* main MD loop */
  printf("Running simulation.\n");
  for (unsigned int step = 0; step <= nsteps; step++) // running simulation loop
  {
    /* first integration for velverlet */
    first_integration(np, dt, box, r_d, v_d, f_d, nthreads);

    /* force calculation */
    force_calculation(np, box, lj1, lj2, r_d, f_d, rcut * rcut, nthreads);

    /* compute temperature and potential */
    compute_info(np, v_d, f_d, info_d, scratch_d, nthreads, nblocks);

    /* second integration for velverlet */
    second_integration(np, dt, v_d, f_d, nthreads);

    /* write output frames and system information, if requested */
    if ((step % nprint) == 0) {
      cudaMemcpy(r, r_d, memSize, cudaMemcpyDeviceToHost);
      cudaMemcpy(info, info_d, 16 * sizeof(float), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy");
      output(traj, step, info, r, np);
      printf("time step %d \n", step);
    }
  }

  gettimeofday(&end, NULL); // get end time
  long timeusr =
      (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
  printf("time is %ld  microseconds\n",
         timeusr); // the spending time on simulation in microseconds

  // free memories and close files
  fclose(traj);
  free(r);
  free(v);
  free(f);
  free(info);
  cudaFree(r_d);
  cudaFree(v_d);
  cudaFree(f_d);
  cudaFree(info_d);
  cudaFree(scratch_d);
  return 0;
}
