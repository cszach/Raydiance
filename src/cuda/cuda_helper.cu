#include "cuda_helper.cuh"
#include <iostream>

void checkCuda(cudaError_t result, char const *const function_name,
               const char *const filename, int const line_num) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << filename << ":" << line_num << " '" << function_name
              << "': " << cudaGetErrorString(result) << "\n";
    cudaDeviceReset();
    exit(99);
  }
}