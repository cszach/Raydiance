#ifndef SCENE_H
#define SCENE_H

#include <memory>
#include <vector>

#include "Object.hpp"

using std::shared_ptr;

class Scene : public Object {
public:
  Scene();

  void clear();

  void add(shared_ptr<Object> object);

  virtual bool hit(const Ray &ray, double t_min, double t_max,
                   HitRecord &rec) const override;

private:
  std::vector<shared_ptr<Object>> objects;
};

#endif // SCENE_H