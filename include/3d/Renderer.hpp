#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.hpp"
#include "Interval.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Vec3.hpp"

class Renderer {
private:
  Camera &_camera;
  int _output_width;
  int _output_height;

  int _num_pixels;
  std::vector<float> _frame_buffer;

  Point3 _pixel00;
  Vec3 _pixel_delta_u;
  Vec3 _pixel_delta_v;
  Point3 _center;

  static const Interval intensity;

public:
  int num_samples = 10;
  int num_bounces = 10;

  Renderer(Camera &camera, int output_width, int output_height);

  Camera &getCamera() const;
  int getOutputWidth() const;
  int getOutputHeight() const;
  int getNumSamples() const;
  const std::vector<float> &getFrameBuffer() const;

  void setCamera(const Camera &camera);
  void setOutputSize(int output_width, int output_height);
  void setNumSamples(int num_samples);

  void render(const Scene &scene);

private:
  Ray getRay(int i, int j) const;
  Point3 getPixelSampleSquare() const;
  Color getRayColor(const Ray &ray, int depth, const Scene &scene) const;
};

const Interval Renderer::intensity(0.0, 0.999);

#endif // RENDERER_H