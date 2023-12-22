#include <cmath>

#include "Camera.hpp"
#include "MathUtils.hpp"

Camera::Camera(double vertical_fov, double aspect_ratio) {
  setVerticalFov(vertical_fov);
  setAspectRatio(aspect_ratio);
}

Camera::Camera() = default;

double Camera::getVerticalFOV() const { return _vertical_fov; }
double Camera::getAspectRatio() const { return _aspect_ratio; }
double Camera::getFocalLength() const { return _focal_length; }
Point3 Camera::getViewportU() const { return _viewport_u; }
Point3 Camera::getViewportV() const { return _viewport_v; }
Point3 Camera::getViewportUpperLeft() const { return _viewport_upper_left; }

void Camera::setVerticalFov(double vertical_fov) {
  _vertical_fov = vertical_fov;
  _theta = degToRad(vertical_fov);
  _h = tan(_theta / 2);

  setFocalLength(_focal_length);
}

void Camera::setAspectRatio(double aspect_ratio) {
  _aspect_ratio = aspect_ratio;

  auto viewport_width = _viewport_height * _aspect_ratio;
  _viewport_u = Vec3(viewport_width, 0, 0);

  computerViewportUpperLeft();
}

void Camera::setFocalLength(double focal_length) {
  _focal_length = focal_length;
  _viewport_height = 2 * _h * _focal_length;

  auto viewport_width = _viewport_height * _aspect_ratio;

  _viewport_u = Vec3(viewport_width, 0, 0);
  _viewport_v = Vec3(0, -_viewport_height, 0);

  computerViewportUpperLeft();
}

bool Camera::hit(const Ray &ray, double t_min, double t_max,
                 HitRecord &rec) const {
  return false;
}

// PRIVATE

void Camera::computerViewportUpperLeft() {
  _viewport_upper_left =
      _position - Vec3(0, 0, _focal_length) - _viewport_u / 2 - _viewport_v / 2;
}
