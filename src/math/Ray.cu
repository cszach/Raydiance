#include "Ray.hpp"

__device__ Ray::Ray() : origin(Point3(0, 0, 0)), direction(Vec3(0, 0, -1)) {}
__device__ Ray::Ray(const Point3 &origin, const Vec3 &direction)
    : origin(origin), direction(direction) {}

__device__ Point3 Ray::at(float t) const { return origin + t * direction; }
