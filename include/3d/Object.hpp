#include "Ray.hpp"

struct HitRecord {
  double t;
};

class Object {
public:
  virtual bool hit(const Ray &r, double t_min, double t_max,
                   HitRecord &rec) const = 0;
};