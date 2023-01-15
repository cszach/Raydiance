#include "Camera.hpp"

#include <cmath>

#include "MathUtils.hpp"

void Camera::setViewportHeight(double viewport_height,
                               bool setViewportWidth = true) {
  this->viewport_height = viewport_height;
  this->vertical = Vec3(0, viewport_height, 0);

  if (setViewportWidth) {
    this->setViewportWidth(this->viewport_height * this->aspect_ratio, false);
    this->computeLowerLeftCorner();
  }
}

void Camera::setViewportWidth(double viewport_width,
                              bool setViewportHeight = true) {
  this->viewport_width = viewport_width;
  this->horizontal = Vec3(viewport_width, 0, 0);

  if (setViewportHeight) {
    this->setViewportHeight(this->viewport_width / this->aspect_ratio, false);
    this->computeLowerLeftCorner();
  }
}

void Camera::computeLowerLeftCorner() {
  this->lower_left_corner = this->position - this->horizontal / 2 -
                            this->vertical / 2 - Vec3(0, 0, focal_length);
}

Camera::Camera() : Camera(50, 1.0) {}
Camera::Camera(double vertical_fov, double aspect_ratio)
    : vertical_fov(vertical_fov), aspect_ratio(aspect_ratio) {
  double theta = degToRad(vertical_fov);
  double h = tan(theta / 2.0);

  this->viewport_height = 2.0 * h;

  this->setViewportHeight(2.0 * h);
  this->computeLowerLeftCorner();
}

double Camera::getVerticalFOV() const { return this->vertical_fov; }
double Camera::getAspectRatio() const { return this->aspect_ratio; }
double Camera::getViewportHeight() const { return this->viewport_height; }
double Camera::getViewportWidth() const { return this->viewport_width; }
double Camera::getFocalLength() const { return this->focal_length; }
Point3 Camera::getPosition() const { return this->position; }

void Camera::setVerticalFov(double vertical_fov) {
  this->vertical_fov = vertical_fov;

  double theta = degToRad(vertical_fov);
  double h = tan(theta / 2.0);

  this->setViewportHeight(2.0 * h);
}

void Camera::setAspectRatio(double aspect_ratio) {
  this->aspect_ratio = aspect_ratio;
  this->viewport_width = this->viewport_height * this->aspect_ratio;
  this->lower_left_corner.setX(
      this->lower_left_corner.getX() +
      (this->horizontal.getX() - this->viewport_width) / 2.0);
  this->horizontal.setX(this->viewport_width);
  this->vertical.setY(this->viewport_height);
}

void Camera::setFocalLength(double focal_length) {
  this->lower_left_corner.setZ(this->lower_left_corner.getZ() +
                               this->focal_length - focal_length);
  this->focal_length = focal_length;
}

void Camera::setPosition(const Point3 &position) {
  this->lower_left_corner += position - this->position;
  this->position = position;
}

Ray Camera::getRay(double u, double v) const {
  return Ray(this->position, this->lower_left_corner + u * this->horizontal +
                                 v * this->vertical - this->position);
}

// PRIVATE
