#ifndef AABB_H
#define AABB_H

#include "Interval.cuh"
#include "Ray.cuh"
#include "Vec3.cuh"

class AABB {
public:
  Interval x, y, z;

  __device__ AABB();
  __device__ AABB(const Interval &_x, const Interval &_y, const Interval &_z);
  __device__ AABB(const Point3 &a, const Point3 &b);
  __device__ AABB(const AABB &a, const AABB &b);

  __device__ const Interval &axis(int i) const;
  __device__ bool hit(const Ray &ray, Interval ray_t) const;
};

#endif // AABB_H