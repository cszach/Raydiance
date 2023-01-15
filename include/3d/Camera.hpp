#ifndef CAMERA_H
#define CAMERA_H

#include <cmath>

#include "Ray.hpp"
#include "Vec3.hpp"

class Camera {
private:
  double vertical_fov = 50;
  double aspect_ratio = 1.0;

  double viewport_height;
  double viewport_width;
  double focal_length = 1.0;
  Point3 position = Point3(0, 0, 0);

  Vec3 horizontal;
  Vec3 vertical;
  Point3 lower_left_corner;

public:
  Camera();
  Camera(double vertical_fov, double aspect_ratio);

  double getVerticalFOV() const;
  double getAspectRatio() const;
  double getViewportHeight() const;
  double getViewportWidth() const;
  double getFocalLength() const;
  Point3 getPosition() const;

  void setVerticalFov(double vertical_fov);

  void setAspectRatio(double aspect_ratio);

  void setFocalLength(double focal_length);

  void setPosition(const Point3 &position);

  Ray getRay(double u, double v) const;

private:
  void setViewportHeight(double viewport_height, bool setViewportWidth);

  void setViewportWidth(double viewport_width, bool setViewportHeight);

  void computeLowerLeftCorner();
};

#endif // CAMERA_H