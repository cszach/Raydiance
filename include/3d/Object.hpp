#ifndef OBJECT_H
#define OBJECT_H

#include <memory>

#include "Material.hpp"
#include "Ray.hpp"
#include "Vec3.hpp"

using std::shared_ptr;

struct HitRecord {
  double t;
  Point3 p;
  Vec3 normal;
  shared_ptr<Material> pMaterial;
  bool front_face;

  inline void setFaceNormal(const Ray &ray, const Vec3 &outward_normal) {
    front_face = dotProduct(ray.getDirection(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class Object {
private:
  Point3 position;

public:
  Object() : position(Point3(0, 0, 0)) {}

  Point3 getPosition() const { return this->position; }
  void setPosition(const Point3 &position) { this->position = position; }

  virtual bool hit(const Ray &ray, double t_min, double t_max,
                   HitRecord &rec) const = 0;
};

#endif // OBJECT_H