#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>

#include "MathUtils.hpp"
#include "Vec3.hpp"

class Camera {
private:
  double vertical_fov = 50;
  double aspect_ratio = 1.0;

  double viewport_height;
  double viewport_width;
  double focal_length = 1.0;
  Point3 origin = Point3(0, 0, 0);

  Vec3 horizontal;
  Vec3 vertical;
  Point3 lower_left_corner;

public:
  Camera() : Camera(50, 1.0) {}
  Camera(double vertical_fov, double aspect_ratio)
      : vertical_fov(vertical_fov), aspect_ratio(aspect_ratio) {
    double theta = degToRad(vertical_fov);
    double h = tan(theta / 2.0);

    this->viewport_height = 2.0 * h;

    this->setViewportHeight(2.0 * h);
    this->computeLowerLeftCorner();
  }

  double getVerticalFOV() const { return this->vertical_fov; }
  double getAspectRatio() const { return this->aspect_ratio; }
  double getViewportHeight() const { return this->viewport_height; }
  double getViewportWidth() const { return this->viewport_width; }
  double getFocalLength() const { return this->focal_length; }
  Point3 getOrigin() const { return this->origin; }

  void setVerticalFov(double vertical_fov) {
    this->vertical_fov = vertical_fov;

    double theta = degToRad(vertical_fov);
    double h = tan(theta / 2.0);

    this->setViewportHeight(2.0 * h);
  }

  void setAspectRatio(double aspect_ratio) {
    this->aspect_ratio = aspect_ratio;
    this->viewport_width = this->viewport_height * this->aspect_ratio;
    this->lower_left_corner.setX(
        this->lower_left_corner.getX() +
        (this->horizontal.getX() - this->viewport_width) / 2.0);
    this->horizontal.setX(this->viewport_width);
    this->vertical.setY(this->viewport_height);
  }

  void setFocalLength(double focal_length) {
    this->lower_left_corner.setZ(this->lower_left_corner.getZ() +
                                 this->focal_length - focal_length);
    this->focal_length = focal_length;
  }

  void setOrigin(Point3 origin) {
    this->lower_left_corner += origin - this->origin;
    this->origin = origin;
  }

private:
  void setViewportHeight(double viewport_height, bool setViewportWidth = true) {
    this->viewport_height = viewport_height;
    this->vertical = Vec3(0, viewport_height, 0);

    if (setViewportWidth) {
      this->setViewportWidth(this->viewport_height * this->aspect_ratio, false);
      this->computeLowerLeftCorner();
    }
  }

  void setViewportWidth(double viewport_width, bool setViewportHeight = true) {
    this->viewport_width = viewport_width;
    this->horizontal = Vec3(viewport_width, 0, 0);

    if (setViewportHeight) {
      this->setViewportHeight(this->viewport_width / this->aspect_ratio, false);
      this->computeLowerLeftCorner();
    }
  }

  void computeLowerLeftCorner() {
    this->lower_left_corner = this->origin - this->horizontal / 2 -
                              this->vertical / 2 - Vec3(0, 0, focal_length);
  }
};

#endif // CAMERA_H