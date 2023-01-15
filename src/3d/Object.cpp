#include "Object.hpp"

#include <memory>

Object::Object() : position(Point3(0, 0, 0)) {}

Point3 Object::getPosition() const { return this->position; }
void Object::setPosition(const Point3 &position) { this->position = position; }