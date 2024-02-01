#ifndef RAY_H
#define RAY_H

#include "Vec3.cuh"

class Ray {
public:
  Point3 origin;
  Vec3 direction;

  __device__ Ray();
  __device__ Ray(const Point3 &origin, const Vec3 &direction);

  __device__ Point3 at(float t) const;
};

#endif // RAY_H
