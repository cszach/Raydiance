#include "Scene.hpp"

Scene::Scene() {}

void Scene::clear() { this->objects.clear(); }

void Scene::add(shared_ptr<Object> object) { this->objects.push_back(object); }

bool Scene::hit(const Ray &ray, double t_min, double t_max,
                HitRecord &rec) const {
  HitRecord temp_record;
  bool hit_anything = false;
  double closest_so_far = t_max;

  for (const auto &object : this->objects) {
    if (object->hit(ray, t_min, t_max, temp_record)) {
      if (temp_record.t < closest_so_far) {
        hit_anything = true;
        closest_so_far = temp_record.t;
        rec = temp_record;
      }
    }
  }

  return hit_anything;
}