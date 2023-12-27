#ifndef OBJECT_H
#define OBJECT_H

#include <memory>

#include "Ray.hpp"
#include "Vec3.hpp"

struct HitRecord {
  float t;
  Point3 p;
  Vec3 normal;
  bool front_face;

  __device__ inline void setFaceNormal(const Ray &ray,
                                       const Vec3 &outward_normal) {
    front_face = ray.direction.dot(outward_normal) < 0.0f;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class Object {
private:
  Point3 _position;

public:
  __device__ Object();
  virtual ~Object() = default;

  __host__ __device__ Point3 getPosition() const;
  __device__ void setPosition(const Point3 &position);

  __device__ virtual bool hit(const Ray &ray, float t_min, float t_max,
                              HitRecord &rec) const = 0;
};

#endif // OBJECT_H
