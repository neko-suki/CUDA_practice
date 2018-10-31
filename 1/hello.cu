#include <stdio.h>
#include "hello_world.hpp"

__global__ void helloFromGPU(){
  printf("Hello World from GPU\n");
}

void launch_cuda(){
  helloFromGPU<<<1, 10>>>();
  cudaDeviceReset();
}
