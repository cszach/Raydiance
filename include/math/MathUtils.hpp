#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <curand_kernel.h>
#include <limits>

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

__device__ inline float degToRad(float deg) { return deg * pi / 180.0; }

__device__ inline float clamp(float value, float min, float max) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

__device__ inline float random_float(curandState *local_rand_state) {
  return curand_uniform(local_rand_state);
}

__device__ inline float random_float(float min, float max,
                                     curandState *local_rand_state) {
  return min + (max - min) * random_float(local_rand_state);
}

#endif // MATH_UTILS_H
