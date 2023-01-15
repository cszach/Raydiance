#include "Sphere.hpp"

#include "Ray.hpp"
#include "Vec3.hpp"

Sphere::Sphere(double radius) : radius(radius) {}

bool Sphere::hit(const Ray &ray, double t_min, double t_max,
                 HitRecord &rec) const {
  Vec3 o_c =
      ray.getOrigin() - this->getPosition(); // ray origin - sphere position
  double a = dotProduct(ray.getDirection(), ray.getDirection());
  double half_b = dotProduct(ray.getDirection(), o_c);
  double c = dotProduct(o_c, o_c) - this->radius * this->radius;

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
  Vec3 outward_normal = (rec.p - this->getPosition()) / radius;
  rec.setFaceNormal(ray, outward_normal);

  return true;
}