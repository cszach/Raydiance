#include <memory>

#include "Object.hpp"

__device__ Object::Object() = default;

__device__ Point3 Object::getPosition() const { return _position; }
__device__ void Object::setPosition(const Point3 &position) {
  _position = position;
}