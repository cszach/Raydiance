#ifndef INTERVAL_H
#define INTERVAL_H

#include "MathUtils.cuh"

class Interval {
public:
  float min;
  float max;

  __host__ __device__ Interval() : min(+infinity), max(-infinity) {}

  __host__ __device__ Interval(float min, float max) : min(min), max(max) {}

  __host__ __device__ bool contains(float x) const {
    return min <= x && x <= max;
  }

  __host__ __device__ bool surrounds(float x) const {
    return min < x && x < max;
  }

  __host__ __device__ float clamp(float x) const {
    if (x < min)
      return min;

    if (x > max)
      return max;

    return x;
  }

  static const Interval empty;
  static const Interval universe;
};

const Interval empty(+infinity, -infinity);
const Interval universe(-infinity, +infinity);

#endif