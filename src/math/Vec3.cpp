#include "Vec3.hpp"

#include <cmath>

Vec3::Vec3() : _x(0), _y(0), _z(0) {}
Vec3::Vec3(double x, double y, double z) : _x(x), _y(y), _z(z) {}

double Vec3::getX() const { return _x; }
double Vec3::getY() const { return _y; }
double Vec3::getZ() const { return _z; }

void Vec3::setX(double x) { _x = x; }
void Vec3::setY(double y) { _y = y; }
void Vec3::setZ(double z) { _z = z; }

double Vec3::length() const { return sqrt(lengthSquared()); }

double Vec3::lengthSquared() const { return _x * _x + _y * _y + _z * _z; }

bool Vec3::equals(const Vec3 &v) const {
  return _x == v.getX() && _y == v.getY() && _z == v.getZ();
}

Vec3 Vec3::operator-() const { return Vec3(-_x, -_y, -_z); }

Vec3 &Vec3::operator+=(const Vec3 &v) {
  _x += v.getX();
  _y += v.getY();
  _z += v.getZ();

  return *this;
}

Vec3 &Vec3::operator-=(const Vec3 &v) {
  _x -= v.getX();
  _y -= v.getY();
  _z -= v.getZ();

  return *this;
}

Vec3 &Vec3::operator*=(const Vec3 &v) {
  _x *= v.getX();
  _y *= v.getY();
  _z *= v.getZ();

  return *this;
}

Vec3 &Vec3::operator/=(const double t) {
  _x /= t;
  _y /= t;
  _z /= t;

  return *this;
}
