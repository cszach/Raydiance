#include "Renderer.hpp"

#include "MathUtils.hpp"

Renderer::Renderer(int output_width, int output_height) {
  this->setOutputSize(output_width, output_height);
}

int Renderer::getOutputWidth() { return this->output_width; }
int Renderer::getOutputHeight() { return this->output_height; }
float *Renderer::getFrameBuffer() { return this->frame_buffer; }

void Renderer::setOutputSize(int output_width, int output_height) {
  this->output_width = output_width;
  this->output_height = output_height;

  this->num_pixels = output_width * output_height;
  this->frame_buffer_size = 3 * num_pixels * sizeof(float);
  this->frame_buffer = (float *)malloc(frame_buffer_size);
}

void Renderer::render(const Scene &scene, const Camera &camera) {
  for (int i = 0; i < this->output_width; i++) {
    for (int j = 0; j < this->output_height; j++) {
      auto u = static_cast<double>(i) / (this->output_width - 1);
      auto v = static_cast<double>(j) / (this->output_height - 1);

      Ray ray = camera.getRay(u, v);
      Color ray_color = this->getRayColor(ray, scene);

      // int r = static_cast<int>(256 * clamp(ray_color.getX(), 0, 0.999));
      // int g = static_cast<int>(256 * clamp(ray_color.getY(), 0, 0.999));
      // int b = static_cast<int>(256 * clamp(ray_color.getZ(), 0, 0.999));

      int pixel_index = 3 * (i + j * this->output_width);

      this->frame_buffer[pixel_index] = clamp(ray_color.getX(), 0, 0.999);
      this->frame_buffer[pixel_index + 1] = clamp(ray_color.getY(), 0, 0.999);
      this->frame_buffer[pixel_index + 2] = clamp(ray_color.getZ(), 0, 0.999);
    }
  }
}

Color Renderer::getRayColor(const Ray &ray, const Scene &scene) {
  HitRecord rec;

  if (scene.hit(ray, 0.001, INFINITY, rec)) {
    return 0.5 * (rec.normal + Color(1, 1, 1));
  }

  return Color(0, 0, 0);
}