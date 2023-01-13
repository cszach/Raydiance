#ifndef RENDERER_H
#define RENDERER_H

#include "Camera.hpp"
#include "ImageOutput.hpp"
#include "Scene.hpp"

using std::shared_ptr;

class Renderer {
private:
  shared_ptr<ImageOutput> output;

public:
  Renderer(shared_ptr<ImageOutput> output) : output(output) {}

  shared_ptr<ImageOutput> getOutput() const { return this->output; }

  void render(const Scene &scene, const Camera &camera) {
    const int WIDTH = (*this->output).getWidth();
    const int HEIGHT = (*this->output).getHeight();

    for (int i = 0; i < WIDTH; i++) {
      for (int j = 0; j < HEIGHT; j++) {
        int r =
            static_cast<int>(255.999 * static_cast<double>(i) / (WIDTH - 1));
        int g =
            static_cast<int>(255.999 * static_cast<double>(j) / (HEIGHT - 1));
        int b = static_cast<int>(255.99 * 0.25);

        (*this->output).writeColor(r, g, b);
      }
    }
  }
};

#endif