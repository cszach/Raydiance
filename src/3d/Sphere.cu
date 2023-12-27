#include "Sphere.hpp"

#include "Ray.hpp"
#include "Vec3.hpp"

__device__ Sphere::Sphere(float radius) : _radius(radius) {}

__device__ bool Sphere::hit(const Ray &ray, float t_min, float t_max,
                            HitRecord &rec) const {
  Vec3 o_c = ray.origin - getPosition(); // ray origin - sphere position
  float a = ray.direction.dot(ray.direction);
  float half_b = ray.direction.dot(o_c);
  float c = o_c.dot(o_c) - _radius * _radius;

  float discriminant = half_b * half_b - a * c;

  // Find the nearest root that lies in the acceptable range

  float sqrt_d = sqrt(discriminant);
  float root = (-half_b - sqrt_d) / a;

  if (root < t_min || root > t_max) {
    root = (-half_b + sqrt_d) / a;
    if (root < t_min || root > t_max) {
      return false;
    }
  }

  rec.t = root;
  rec.p = ray.at(root);
  Vec3 outward_normal = (rec.p - getPosition()) / _radius;
  rec.setFaceNormal(ray, outward_normal);

  return true;
}
