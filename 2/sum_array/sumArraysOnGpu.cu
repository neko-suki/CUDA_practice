#include <stdio.h>

__global__ void sumArraysOnGpu(const float *a, const float *b, float *c){
  const size_t i = threadIdx.x;
  
  c[i] = a[i] + b[i];
}

void launch_cuda(const size_t n, const size_t nBytes, const float * a, const float * b, float * c){
  float *d_A, *d_B, *d_C;
  cudaMalloc((float**) &d_A, nBytes);
  cudaMalloc((float**) &d_B, nBytes);
  cudaMalloc((float**) &d_C, nBytes);
  
  cudaMemcpy(d_A, a, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, b, nBytes, cudaMemcpyHostToDevice);
  
  sumArraysOnGpu<<<1, n>>>(d_A, d_B, d_C);

  cudaMemcpy(c, d_C, nBytes, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
