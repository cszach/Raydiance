#ifndef INTERVAL_H
#define INTERVAL_H

#include "MathUtils.cuh"

class Interval {
public:
  float min;
  float max;

  __device__ Interval() : min(+infinity), max(-infinity) {}
  __device__ Interval(float min, float max) : min(min), max(max) {}
  __device__ Interval(const Interval &a, const Interval &b)
      : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}

  __device__ bool contains(float x) const { return min <= x && x <= max; }

  __device__ bool surrounds(float x) const { return min < x && x < max; }

  __device__ float clamp(float x) const {
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