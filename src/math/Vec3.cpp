#include "Vec3.hpp"

#include <cmath>

Vec3::Vec3() : x(0), y(0), z(0) {}
Vec3::Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

void Vec3::set(double x, double y, double z) {
  this->x = x;
  this->y = y;
  this->z = z;
}

double Vec3::length() const { return sqrt(lengthSquared()); }
double Vec3::lengthSquared() const { return x * x + y * y + z * z; }
bool Vec3::equals(const Vec3 &v) const {
  return x == v.x && y == v.y && z == v.z;
}

Vec3 Vec3::operator-() const { return Vec3(-x, -y, -z); }

Vec3 &Vec3::operator+=(const Vec3 &v) {
  x += v.x;
  y += v.y;
  z += v.z;

  return *this;
}

Vec3 &Vec3::operator-=(const Vec3 &v) {
  x -= v.x;
  y -= v.y;
  z -= v.z;

  return *this;
}

Vec3 &Vec3::operator*=(const Vec3 &v) {
  x *= v.x;
  y *= v.y;
  z *= v.z;

  return *this;
}

Vec3 &Vec3::operator/=(const double t) {
  x /= t;
  y /= t;
  z /= t;

  return *this;
}
