#ifndef CAMERA_H
#define CAMERA_H

#include "Object.hpp"
#include "Ray.hpp"
#include "Vec3.hpp"

class Camera : public Object {
private:
  double _vertical_fov;
  double _aspect_ratio;

  double _focal_length = 1.0;
  Point3 _position = Point3(0, 0, 0);

  double _theta;
  double _h;
  double _viewport_height;
  Vec3 _viewport_u;
  Vec3 _viewport_v;
  Point3 _viewport_upper_left;

public:
  Camera(double vertical_fov, double aspect_ratio);
  Camera();

  double getVerticalFOV() const;
  double getAspectRatio() const;
  double getFocalLength() const;
  Vec3 getViewportU() const;
  Vec3 getViewportV() const;
  Point3 getViewportUpperLeft() const;

  void setVerticalFov(double vertical_fov);
  void setAspectRatio(double aspect_ratio);
  void setFocalLength(double focal_length);
  bool hit(const Ray &ray, double t_min, double t_max,
           HitRecord &rec) const override;

private:
  void computerViewportUpperLeft();
};

#endif // CAMERA_H
