#include "Sphere.hpp"

#include "Ray.hpp"
#include "Vec3.hpp"

Sphere::Sphere(double radius) : _radius(radius) {}

bool Sphere::hit(const Ray &ray, double t_min, double t_max,
                 HitRecord &rec) const {
  Vec3 o_c = ray.getOrigin() - getPosition(); // ray origin - sphere position
  double a = dotProduct(ray.getDirection(), ray.getDirection());
  double half_b = dotProduct(ray.getDirection(), o_c);
  double c = dotProduct(o_c, o_c) - _radius * _radius;

  double discriminant = half_b * half_b - a * c;

  // Find the nearest root that lies in the acceptable range

  double sqrt_d = sqrt(discriminant);
  double root = (-half_b - sqrt_d) / a;

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
