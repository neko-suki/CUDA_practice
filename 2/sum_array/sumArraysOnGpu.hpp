#ifndef SUM_ARRAYS_ON_GPU
#define SUM_ARRAYS_ON_GPU
void launch_cuda(const size_t n, const size_t nBytes, const float * a, const float * b, float * c);

#endif // SUM_ARRAYS_ON_GPU
