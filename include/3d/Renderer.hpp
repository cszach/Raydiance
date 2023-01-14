#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.hpp"
#include "ImageOutput.hpp"
#include "MathUtils.hpp"
#include "Ray.hpp"
#include "Scene.hpp"
#include "Vec3.hpp"

class Renderer {
private:
  ImageOutput &output;

public:
  Renderer(ImageOutput &output) : output(output) {}

  ImageOutput &getOutput() const { return this->output; }

  void render(const Scene &scene, const Camera &camera) {
    const int WIDTH = this->output.getWidth();
    const int HEIGHT = this->output.getHeight();

    for (int i = 0; i < WIDTH; i++) {
      for (int j = 0; j < HEIGHT; j++) {
        auto u = static_cast<double>(i) / (WIDTH - 1);
        auto v = static_cast<double>(j) / (HEIGHT - 1);

        Ray ray = camera.getRay(u, v);
        Color ray_color = this->getRayColor(ray, scene);

        int r = static_cast<int>(256 * clamp(ray_color.getX(), 0, 0.999));
        int g = static_cast<int>(256 * clamp(ray_color.getY(), 0, 0.999));
        int b = static_cast<int>(256 * clamp(ray_color.getZ(), 0, 0.999));

        this->output.writeColor(r, g, b);
      }
    }
  }

private:
  Color getRayColor(const Ray &ray, const Scene &scene) {
    HitRecord rec;

    if (scene.hit(ray, 0.001, INFINITY, rec)) {
      return 0.5 * (rec.normal + Color(1, 1, 1));
    }

    return Color(0, 0, 0);
  }
};

#endif