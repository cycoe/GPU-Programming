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

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(-1);
  }
}

__host__ __device__ float
pbc(float x, float box_len) // implement periodic bondary condition
{
  float box_half = box_len * 0.5;
  if (x > box_half)
    x -= box_len;
  else if (x < -box_half)
    x += box_len;
  return x;
}

// randome number generator [0.0-1.0)
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
        if (r < min_dis) // a minimum safe distance to avoid the overlap of LJ
                         // particles
        {
          find_pos = false;
          break;
        }
      }
    }
    if (R2S() > 0.5) // randomly generate the type of particle, 1.0 represent
                     // type A and 2.0 represent type B
      ri.w = 1.0;
    else
      ri.w = 2.0;

    r[i] = ri;
    v[i].w = 1.0;
  }
}

// first step integration of velocity verlet algorithm
extern "C" __global__ void first_integration_kernel(unsigned int np, float dt,
                                                    float3 box, float4 *r,
                                                    float4 *v, float4 *f) {
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
                       float4 *v, float4 *f, unsigned int block_size) {

  dim3 grid((np / block_size) + 1, 1, 1);
  dim3 block(block_size, 1, 1);

  first_integration_kernel<<<grid, block>>>(np, dt, box, r, v, f);
  // block until the device has completed
  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  // check if kernel execution generated an error
  checkCUDAError("kernel execution");
}

// non-bonded force calculation
extern "C" __global__ void force_calculation_kernel(unsigned int np, float3 box,
                                                    float3 lj1, float3 lj2,
                                                    float4 *r, float4 *f,
                                                    float rcutsq) {
  extern __shared__ float4 spos[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float4 force = make_float4(0.0, 0.0, 0.0, 0.0);
  float4 ri = make_float4(0.0, 0.0, 0.0, 0.0);
  if (i < np)
    ri = r[i];

  for (int start = 0; start < np; start += blockDim.x) {
    // load data
    float4 posj = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (start + threadIdx.x < np)
      posj = r[start + threadIdx.x];
    __syncthreads();

    spos[threadIdx.x] = posj;
    __syncthreads();

    int end_offset = blockDim.x;
    end_offset = min(end_offset, np - start);

    if (i < np) {
      for (int cur_offset = 0; cur_offset < end_offset; cur_offset++) {
        float4 rj = spos[cur_offset];
        int j = start + cur_offset;

        /* particles have no interactions with themselves */
        if (i == j)
          continue;

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

          float r2inv = float(1.0) / rsq;
          float r6inv = r2inv * r2inv * r2inv;

          float ffac = r2inv * r6inv *
                       (float(12.0) * lj1_ij * r6inv -
                        float(6.0) * lj2_ij); // force between particle i and j
          float epot = r6inv * (lj1_ij * r6inv -
                                lj2_ij); // potential between particle i and j

          force.x += ffac * dx;
          force.y += ffac * dy;
          force.z += ffac * dz;
          force.w += epot;
        }
      }
    }
  }

  if (i < np) {
    f[i] = force;
    //	printf("%d  %f   %f   %f   %f  \n", i, force.x,  force.y, force.z,
    //force.w);
  }
}

void force_calculation(unsigned int np, float3 box, float3 epsilon,
                       float3 sigma, float4 *r, float4 *f, float rcut,
                       unsigned int block_size) {

  dim3 grid((np / block_size) + 1, 1, 1);
  dim3 block(block_size, 1, 1);
  unsigned int shared_bytes = sizeof(float4) * block_size;
  force_calculation_kernel<<<grid, block, shared_bytes>>>(np, box, epsilon,
                                                          sigma, r, f, rcut);
  // block until the device has completed
  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  // check if kernel execution generated an error
  checkCUDAError("kernel execution");
}

// second step integration of velocity verlet algorithm
extern "C" __global__ void second_integration_kernel(unsigned int np, float dt,
                                                     float4 *v, float4 *f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < np) {
    float mass = v[i].w;
    v[i].x += 0.5 * dt * f[i].x / mass;
    v[i].y += 0.5 * dt * f[i].y / mass;
    v[i].z += 0.5 * dt * f[i].z / mass;
  }
}

void second_integration(unsigned int np, float dt, float4 *v, float4 *f,
                        unsigned int block_size) {

  dim3 grid((np / block_size) + 1, 1, 1);
  dim3 block(block_size, 1, 1);

  second_integration_kernel<<<grid, block>>>(np, dt, v, f);
  // block until the device has completed
  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  // check if kernel execution generated an error
  checkCUDAError("kernel execution");
}

// system information collection for temperature, kinetic energy, potential and
// total energy

__global__ void compute_info_sums_kernel(unsigned int np, float4 *v, float4 *f,
                                         float2 *scratch) {
  extern __shared__ float2 sdata[];
  int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  float2 tempo = make_float2(0.0, 0.0);
  if (i < np) {
    float4 vi = v[i];
    float mass = vi.w;
    tempo.x = mass * (vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
    tempo.y = f[i].w;

    if (i + blockDim.x < np) {
      vi = v[i + blockDim.x];
      mass = vi.w;
      tempo.x += mass * (vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
      tempo.y += f[i + blockDim.x].w;
    }
  }

  sdata[threadIdx.x] = tempo;
  __syncthreads();

  int offs = blockDim.x >> 1;
  while (offs > 0) {
    if (threadIdx.x < offs) {
      sdata[threadIdx.x].x += sdata[threadIdx.x + offs].x;
      sdata[threadIdx.x].y += sdata[threadIdx.x + offs].y;
    }
    offs >>= 1;
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    scratch[blockIdx.x].x = sdata[0].x;
    scratch[blockIdx.x].y = sdata[0].y;
  }
}

__global__ void compute_info_final_kernel(unsigned int np, float *info,
                                          float2 *scratch,
                                          unsigned int num_partial_sums) {
  extern __shared__ float2 sdata[];
  float2 final_sum = make_float2(0.0, 0.0);
  for (int start = 0; start < num_partial_sums; start += blockDim.x * 2) {
    float2 tempo = make_float2(0.0, 0.0);
    if (start + threadIdx.x < num_partial_sums) {
      float2 scr = scratch[start + threadIdx.x];
      tempo.x = scr.x;
      tempo.y = scr.y;
      if (start + threadIdx.x + blockDim.x < num_partial_sums) {
        scr = scratch[start + threadIdx.x + blockDim.x];
        tempo.x += scr.x;
        tempo.y += scr.y;
      }
    }

    sdata[threadIdx.x] = tempo;
    __syncthreads();

    int offs = blockDim.x >> 1;
    while (offs > 0) {
      if (threadIdx.x < offs) {
        sdata[threadIdx.x].x += sdata[threadIdx.x + offs].x;
        sdata[threadIdx.x].y += sdata[threadIdx.x + offs].y;
      }
      offs >>= 1;
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
    float temp = 2.0 * ekin / float(nfreedom);
    float energy = ekin + potential;

    info[0] = temp;
    info[1] = potential;
    info[2] = energy;
  }
}

void compute_info(unsigned int np, float4 *v, float4 *f, float2 *scratch,
                  float *info, unsigned int block_size) {

  unsigned int n_blocks = (int)ceil((float)np / (float)block_size);
  dim3 grid(n_blocks, 1, 1);
  dim3 threads(block_size / 2, 1, 1);
  unsigned int shared_bytes = sizeof(float2) * block_size / 2;

  compute_info_sums_kernel<<<grid, threads, shared_bytes>>>(np, v, f, scratch);
  // block until the device has completed
  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  // check if kernel execution generated an error
  checkCUDAError("kernel execution");

  int final_block_size = 512;
  grid = dim3(1, 1, 1);
  threads = dim3(final_block_size, 1, 1);
  shared_bytes = sizeof(float2) * final_block_size;

  compute_info_final_kernel<<<grid, threads, shared_bytes>>>(np, info, scratch,
                                                             n_blocks);
  // block until the device has completed
  // cudaThreadSynchronize();
  cudaDeviceSynchronize();

  // check if kernel execution generated an error
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
  /* phase seperation parameters for video:
  unsigned int nsteps = 200000;
  unsigned int nprint = 5000;
  float3 epsilon = make_float3(1.0, 0.2, 1.0);
  */
  // running parameters
  unsigned int np = 2700;    // the number of particles
  unsigned int nsteps = 500; // the number of time steps
  float dt = 0.001;          // integration time step
  float rcut = 3.0;          // the cutoff radius of interactions

  unsigned int nprint = 100;     //  period for data output
  unsigned int block_size = 128; //  the number of threads in a block
  timeval start;                 // start time
  timeval end;                   // end time

  float3 box =
      make_float3(15.0, 15.0, 15.0); // box size in x, y, and z directions
  float3 epsilon = make_float3(
      1.0, 0.5, 1.0); // epsilon.x for type 1.0 and 1.0; epsilon.y for type 1.0
                      // and 2.0; epsilon.z for type 1.0 and 2.0
  float3 sigma = make_float3(
      1.0, 1.0, 1.0); // sigma.x for type 1.0 and 1.0; sigma.y for type 1.0
                      // and 2.0; sigma.z for type 1.0 and 2.0
  float min_dis =
      sigma.x *
      0.9; // the minimum distance between particles for system generation
  float3 lj1, lj2;

  lj1.x = 4.0 * epsilon.x * pow(sigma.x, int(12));
  lj1.y = 4.0 * epsilon.y * pow(sigma.y, int(12));
  lj1.z = 4.0 * epsilon.z * pow(sigma.z, int(12));

  lj2.x = 4.0 * epsilon.x * pow(sigma.x, int(6));
  lj2.y = 4.0 * epsilon.y * pow(sigma.y, int(6));
  lj2.z = 4.0 * epsilon.z * pow(sigma.z, int(6));

  // host memory allocation
  float4 *h_r =
      (float4 *)malloc(np * sizeof(float4)); // rx, ry, rz, type(0, 1, 2 ...)
  float4 *h_v = (float4 *)malloc(np * sizeof(float4)); // vx, vy, vz, mass
  float4 *h_f = (float4 *)malloc(np * sizeof(float4)); // fx, fy, fz, potential
  float *h_info =
      (float *)malloc(16 * sizeof(float)); // temperature, potential, energy ...

  // device memory allocation
  float4 *d_r;
  float4 *d_v;
  float4 *d_f;
  float *d_info;
  float2 *d_scratch;
  cudaMalloc((void **)&d_r,
             np * sizeof(float4)); // rx, ry, rz, type(0, 1, 2 ...)
  cudaMalloc((void **)&d_v, np * sizeof(float4)); // vx, vy, vz, mass
  cudaMalloc((void **)&d_f, np * sizeof(float4)); // fx, fy, fz, potential
  cudaMalloc((void **)&d_info,
             16 * sizeof(float)); // temperature, potential, energy ...
  cudaMalloc((void **)&d_scratch,
             (np / block_size + 1) * sizeof(float2)); // temporary data ...

  FILE *traj = fopen(
      "traj.xyz", "w"); // trajectory file in XYZ format that can be open by VMD

  /* generate system information */

  printf("Starting simulation with %d atoms for %d steps.\n", np, nsteps);
  printf("Generating system.\n", np, nsteps);
  init(np, h_r, h_v, box, min_dis);

  cudaMemcpy(d_r, h_r, np * sizeof(float4), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, np * sizeof(float4), cudaMemcpyHostToDevice);

  gettimeofday(&start, NULL); // get start time
  /* main MD loop */
  printf("Running simulation.\n", np, nsteps);
  for (unsigned int step = 0; step <= nsteps; step++) // running simulation loop
  {
    /* first integration for velverlet */
    first_integration(np, dt, box, d_r, d_v, d_f, block_size);

    /* force calculation */
    force_calculation(np, box, lj1, lj2, d_r, d_f, rcut * rcut, block_size);

    /* compute temperature and potential */
    compute_info(np, d_v, d_f, d_scratch, d_info, block_size);

    /* second integration for velverlet */
    second_integration(np, dt, d_v, d_f, block_size);

    /* write output frames and system information, if requested */
    if ((step % nprint) == 0) {
      cudaMemcpy(h_r, d_r, np * sizeof(float4), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_info, d_info, 16 * sizeof(float), cudaMemcpyDeviceToHost);

      output(traj, step, h_info, h_r, np);
      printf("time step %d \n", step);
    }
  }

  gettimeofday(&end, NULL); // get end time
  long timeusr =
      (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
  printf("time is %ld  microseconds\n",
         timeusr); // the spending time on simulation in microseconds

  fclose(traj);
  free(h_r);
  free(h_v);
  free(h_f);
  free(h_info);

  cudaFree(d_r);
  cudaFree(d_v);
  cudaFree(d_f);
  cudaFree(d_info);
  return 0;
}
