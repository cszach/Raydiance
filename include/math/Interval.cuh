#ifndef INTERVAL_H
#define INTERVAL_H

#include "MathUtils.cuh"

class Interval {
public:
  float _min;
  float _max;

  __host__ __device__ Interval() : _min(+infinity), _max(-infinity) {}

  __host__ __device__ Interval(float min, float max) : _min(min), _max(max) {}

  __host__ __device__ bool contains(float x) const {
    return _min <= x && x <= _max;
  }

  __host__ __device__ bool surrounds(float x) const {
    return _min < x && x < _max;
  }

  __host__ __device__ float clamp(float x) const {
    if (x < _min)
      return _min;

    if (x > _max)
      return _max;

    return x;
  }

  static const Interval empty;
  static const Interval universe;
};

const Interval empty(+infinity, -infinity);
const Interval universe(-infinity, +infinity);

#endif