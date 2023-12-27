#include "Vec3.hpp"

#include <cmath>

__host__ __device__ Vec3::Vec3() : x(0), y(0), z(0) {}
__host__ __device__ Vec3::Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

__host__ __device__ void Vec3::set(float x, float y, float z) {
  this->x = x;
  this->y = y;
  this->z = z;
}

__host__ __device__ float Vec3::length() const { return sqrt(lengthSquared()); }
__host__ __device__ float Vec3::lengthSquared() const {
  return x * x + y * y + z * z;
}
__host__ __device__ bool Vec3::equals(const Vec3 &v) const {
  return x == v.x && y == v.y && z == v.z;
}

__host__ __device__ Vec3 Vec3::operator-() const { return Vec3(-x, -y, -z); }

__host__ __device__ Vec3 &Vec3::operator+=(const Vec3 &v) {
  x += v.x;
  y += v.y;
  z += v.z;

  return *this;
}

__host__ __device__ Vec3 &Vec3::operator-=(const Vec3 &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;

  return *this;
}

__host__ __device__ Vec3 &Vec3::operator*=(const Vec3 &v) {
  x *= v.x;
  y *= v.y;
  z *= v.z;

  return *this;
}

__host__ __device__ Vec3 &Vec3::operator/=(const float t) {
  x /= t;
  y /= t;
  z /= t;

  return *this;
}
