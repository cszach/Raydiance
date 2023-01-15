#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Vec3.hpp"

class Renderer {
private:
  int output_width;
  int output_height;
  int num_pixels;
  size_t frame_buffer_size;
  float *frame_buffer;

public:
  Renderer(int output_width, int output_height);

  int getOutputWidth();
  int getOutputHeight();
  float *getFrameBuffer();

  void setOutputSize(int output_width, int output_height);

  void render(const Scene &scene, const Camera &camera);

private:
  Color getRayColor(const Ray &ray, const Scene &scene);
};

#endif // RENDERER_H