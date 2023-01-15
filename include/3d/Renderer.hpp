#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.hpp"
#include "ImageOutput.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Vec3.hpp"

class Renderer {
private:
  ImageOutput &output;

public:
  Renderer(ImageOutput &output);

  ImageOutput &getOutput() const;

  void render(const Scene &scene, const Camera &camera);

private:
  Color getRayColor(const Ray &ray, const Scene &scene);
};

#endif // RENDERER_H