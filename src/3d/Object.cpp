#include <memory>

#include "Object.hpp"

Object::Object() = default;

Point3 Object::getPosition() const { return _position; }
void Object::setPosition(const Point3 &position) { _position = position; }