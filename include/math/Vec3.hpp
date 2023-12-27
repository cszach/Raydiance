#ifndef VEC3_H
#define VEC3_H

#include <cmath>

#include "MathUtils.hpp"

#define RANDVEC3                                                               \
  Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),     \
       curand_uniform(local_rand_state))

class Vec3 {
public:
  float x;
  float y;
  float z;

  __host__ __device__ Vec3();
  __host__ __device__ Vec3(float x, float y, float z);

  __host__ __device__ void set(float x, float y, float z);

  __host__ __device__ float length() const;
  __host__ __device__ float lengthSquared() const;
  __host__ __device__ bool equals(const Vec3 &v) const;

  __host__ __device__ Vec3 operator-() const;
  __host__ __device__ Vec3 &operator+=(const Vec3 &v);
  __host__ __device__ Vec3 &operator-=(const Vec3 &v);
  __host__ __device__ Vec3 &operator*=(const Vec3 &v);
  __host__ __device__ Vec3 &operator/=(const float t);

  __host__ __device__ Vec3 normalize() const;
  __host__ __device__ float dot(const Vec3 &v) const;
  __host__ __device__ Vec3 cross(const Vec3 &v) const;

  __device__ static Vec3 randomInUnitSphere(curandState *local_rand_state);
  __device__ static Vec3 randomUnit(curandState *local_rand_state);
};

__host__ __device__ inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x - v.x, u.y - v.y, u.z - v.z);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
  return Vec3(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ inline Vec3 operator*(const Vec3 &v, const float t) {
  return Vec3(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ inline Vec3 operator*(const float t, const Vec3 &v) {
  return v * t;
}

__host__ __device__ inline Vec3 operator/(const Vec3 &v, const float t) {
  return v * (1 / t);
}

__host__ __device__ inline Vec3 Vec3::normalize() const {
  float l = length();

  return Vec3(x / l, y / l, z / l);
}

__host__ __device__ inline float Vec3::dot(const Vec3 &v) const {
  return x * v.x + y * v.y + z * v.z;
}

__host__ __device__ inline Vec3 Vec3::cross(const Vec3 &v) const {
  return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}

__device__ inline Vec3 Vec3::randomInUnitSphere(curandState *local_rand_state) {
  while (true) {
    auto v = 2.0f * RANDVEC3 - Vec3(1, 1, 1);

    if (v.lengthSquared() < 1) {
      return v;
    }
  }
}

__device__ inline Vec3 Vec3::randomUnit(curandState *local_rand_state) {
  return randomInUnitSphere(local_rand_state).normalize();
}

using Point3 = Vec3;
using Color = Vec3;

#endif // VEC3_H
