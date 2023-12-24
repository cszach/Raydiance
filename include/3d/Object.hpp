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
    front_face = ray.getDirection().dot(outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class Object {
private:
  Point3 _position;

public:
  Object();
  virtual ~Object() = default;

  Point3 getPosition() const;
  void setPosition(const Point3 &position);

  virtual bool hit(const Ray &ray, double t_min, double t_max,
                   HitRecord &rec) const = 0;
};

#endif // OBJECT_H
