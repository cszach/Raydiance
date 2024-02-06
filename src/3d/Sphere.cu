#include "Object.cuh"
#include "Ray.cuh"
#include "Sphere.cuh"
#include "Vec3.cuh"
#include <cstdio>

__device__ Sphere::Sphere(float _radius, Material *_material)
    : radius(_radius) {
  material = _material;
}

__device__ bool Sphere::hit(const Ray &ray, Interval ray_t,
                            HitRecord &rec) const {
  Vec3 o_c = ray.origin - position; // ray origin - sphere position
  float a = ray.direction.dot(ray.direction);
  float half_b = ray.direction.dot(o_c);
  float c = o_c.dot(o_c) - radius * radius;

  float discriminant = half_b * half_b - a * c;

  // Find the nearest root that lies in the acceptable range

  float sqrt_d = sqrt(discriminant);
  float root = (-half_b - sqrt_d) / a;

  if (!ray_t.surrounds(root)) {
    root = (-half_b + sqrt_d) / a;
    if (!ray_t.surrounds(root)) {
      return false;
    }
  }

  rec.t = root;
  rec.p = ray.at(root);
  Vec3 outward_normal = (rec.p - position) / radius;
  rec.setFaceNormal(ray, outward_normal);
  rec.material = material;

  return true;
}

__device__ void Sphere::computeBoundingBox() {
  Vec3 halfExtents = Vec3(radius, radius, radius);
  boundingBox = AABB(position - halfExtents, position + halfExtents);
}
