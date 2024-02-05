#include "AABB.cuh"
#include "Interval.cuh"
#include "Vec3.cuh"

__device__ AABB::AABB() {}
__device__ AABB::AABB(const Interval &_x, const Interval &_y,
                      const Interval &_z)
    : x(_x), y(_y), z(_z) {}
__device__ AABB::AABB(const Point3 &a, const Point3 &b) {
  x = Interval(fmin(a.x, b.x), fmax(a.x, b.x));
  y = Interval(fmin(a.y, b.y), fmax(a.y, b.y));
  z = Interval(fmin(a.z, b.z), fmax(a.z, b.z));
}

__device__ const Interval &AABB::axis(int i) const {
  if (i == 0)
    return x;
  if (i == 1)
    return y;
  /* else */ return z;
}

__device__ bool AABB::hit(const Ray &r, Interval ray_t) const {
  // for (int a = 0; a < 3; a++) {
  //   auto t0 = fmin((axis(a).min - r.origin.get(a)) / r.direction.get(a),
  //                  (axis(a).max - r.origin.get(a)) / r.direction.get(a));
  //   auto t1 = fmax((axis(a).min - r.origin.get(a)) / r.direction.get(a),
  //                  (axis(a).max - r.origin.get(a)) / r.direction.get(a));

  //   ray_t.min = fmax(t0, ray_t.min);
  //   ray_t.max = fmin(t1, ray_t.max);

  //   if (ray_t.max <= ray_t.min)
  //     return false;

  //   return true;
  // }

  for (int a = 0; a < 3; a++) {
    float inverseDirection = 1.0f / r.direction.get(a);
    float origin = r.origin.get(a);

    float t0 = (axis(a).min - origin) * inverseDirection;
    float t1 = (axis(a).max - origin) * inverseDirection;

    if (inverseDirection < 0) {
      float temp = t0;
      t0 = t1;
      t1 = temp;
    }

    if (t0 > ray_t.min)
      ray_t.min = t0;
    if (t1 < ray_t.max)
      ray_t.max = t1;

    if (ray_t.max <= ray_t.min) {
      return false;
    }
  }

  return true;
}