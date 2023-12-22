#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Vec3.hpp"

class Renderer {
private:
  int _output_width;
  int _output_height;

  int _num_pixels;
  std::vector<float> _frame_buffer;

public:
  Renderer(int output_width, int output_height);

  int getOutputWidth() const;
  int getOutputHeight() const;
  std::vector<float> getFrameBuffer() const;

  void setOutputSize(int output_width, int output_height);

  void render(const Scene &scene, const Camera &camera);

private:
  Color getRayColor(const Ray &ray, const Scene &scene) const;
};

#endif // RENDERER_H
