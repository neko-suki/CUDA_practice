#include <iostream>
#include <ctime>
#include <cmath>
#include "sumArraysOnGpu.hpp"

void sumArraysOnHost(const int N, const float *a, const float *b, float *c){
  for(int idx = 0;idx < N;idx++){
    c[idx] = a[idx] + b[idx];
  }
}

void initialData(const size_t size, float *ip){
  time_t t;
  srand((unsigned int) time(&t));
  for(size_t i = 0;i < size;i++){
    ip[i] = static_cast<float>(rand() & 0xff) / 10.0f;
  }
  return;
}

bool check_diff(const size_t n, const float *h_C, const float *d_C){
  constexpr float eps = 1e-5;
  for(size_t i = 0;i < n;++i){
    if (std::fabs(h_C[i] - d_C[i]) > eps){
      std::cout << "different result at " << i << " " << h_C[i] <<" " 
		<< d_C[i] << " " << std::fabs(h_C[i]-d_C[i]) << std::endl;
      return false;
    }
  }
  return true;
}

int main(){
  const size_t nElem = 1024;
  const size_t nBytes = nElem * sizeof(float);

  float *h_A, *h_B, *h_C;
  h_A = new float[nBytes];
  h_B = new float[nBytes];
  h_C = new float[nBytes];

  initialData(nElem, h_A);
  initialData(nElem, h_B);
  
  sumArraysOnHost(nElem, h_A, h_B, h_C);

  float *d_C;
  d_C = new float[nBytes];
  launch_cuda(nElem, nBytes, h_A,h_B, d_C);

  bool is_same = check_diff(nElem, h_C, d_C);
  std::cout <<"same result? " << is_same << std::endl;
  
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return (0);
}
