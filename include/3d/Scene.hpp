#ifndef SCENE_H
#define SCENE_H

#include <memory>
#include <vector>

#include "Object.hpp"

using std::make_shared;
using std::shared_ptr;

class Scene {
public:
  Scene() {}

  void clear() { this->objects.clear(); }

  void add(shared_ptr<Object> object) { this->objects.push_back(object); }

private:
  std::vector<shared_ptr<Object>> objects;
};

#endif // SCENE_H