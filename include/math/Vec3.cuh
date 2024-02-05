#ifndef VEC3_H
#define VEC3_H

#include <cmath>

#include "MathUtils.cuh"

#define RANDVEC3                                                               \
  Vec3(curand_uniform(localRandState), curand_uniform(localRandState),         \
       curand_uniform(localRandState))

class Vec3 {
public:
  float x;
  float y;
  float z;

  __device__ Vec3();
  __device__ Vec3(float x, float y, float z);

  __device__ void set(float x, float y, float z);

  __device__ float length() const;
  __device__ float lengthSquared() const;
  __device__ bool equals(const Vec3 &v) const;
  __device__ bool isNearZero() const;

  __device__ Vec3 operator-() const;
  __device__ Vec3 &operator+=(const Vec3 &v);
  __device__ Vec3 &operator-=(const Vec3 &v);
  __device__ Vec3 &operator*=(const Vec3 &v);
  __device__ Vec3 &operator/=(const float t);

  __device__ Vec3 normalize() const;
  __device__ float dot(const Vec3 &v) const;
  __device__ Vec3 cross(const Vec3 &v) const;

  __device__ static Vec3 randomInUnitSphere(curandState *localRandState);
  __device__ static Vec3 randomUnit(curandState *localRandState);
};

__device__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__device__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__device__ inline Vec3 operator*(const Vec3 &v, const float t) {
  return Vec3(v.x * t, v.y * t, v.z * t);
}

__device__ inline Vec3 operator*(const float t, const Vec3 &v) { return v * t; }

__device__ inline Vec3 operator/(const Vec3 &v, const float t) {
  return v * (1 / t);
}

__device__ inline Vec3 Vec3::normalize() const {
  float l = length();

  return Vec3(x / l, y / l, z / l);
}

__device__ inline float Vec3::dot(const Vec3 &v) const {
  return x * v.x + y * v.y + z * v.z;
}

__device__ inline Vec3 Vec3::cross(const Vec3 &v) const {
  return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}

__device__ inline Vec3 Vec3::randomInUnitSphere(curandState *localRandState) {
  while (true) {
    auto v = 2.0f * RANDVEC3 - Vec3(1, 1, 1);

    if (v.lengthSquared() < 1) {
      return v;
    }
  }
}

__device__ inline Vec3 Vec3::randomUnit(curandState *localRandState) {
  return randomInUnitSphere(localRandState).normalize();
}

using Point3 = Vec3;
using Color = Vec3;

#endif // VEC3_H
