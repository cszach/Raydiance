#ifndef checkCudaError
#define checkCudaError(result) checkCuda((result), #result, __FILE__, __LINE__)
#endif

#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

void checkCuda(cudaError_t result, char const *const function_name,
               const char *const filename, int const line_num);

#endif // CUDA_HELPER_H
