#include "Renderer.hpp"
#include "MathUtils.hpp"

Renderer::Renderer(int output_width, int output_height) {
  setOutputSize(output_width, output_height);
}

int Renderer::getOutputWidth() const { return _output_width; }
int Renderer::getOutputHeight() const { return _output_height; }
std::vector<float> Renderer::getFrameBuffer() const { return _frame_buffer; }

void Renderer::setOutputSize(int output_width, int output_height) {
  _output_width = output_width;
  _output_height = output_height;

  _num_pixels = output_width * output_height;
  _frame_buffer = std::vector<float>(3 * _num_pixels);
}

void Renderer::render(const Scene &scene, const Camera &camera) {
  auto viewport_u = camera.getViewportU();
  auto viewport_v = camera.getViewportV();

  auto pixel_delta_u = viewport_u / _output_width;
  auto pixel_delta_v = viewport_v / _output_height;

  auto pixel00 =
      camera.getViewportUpperLeft() + 0.5 * (pixel_delta_u + pixel_delta_v);

  auto center = camera.getPosition();

  for (int j = 0; j < this->_output_height; ++j) {
    for (int i = 0; i < this->_output_width; ++i) {
      auto pixel_center = pixel00 + (i * pixel_delta_u) + (j * pixel_delta_v);

      Ray ray(center, pixel_center - center);
      Color ray_color = this->getRayColor(ray, scene);

      int pixel_index = 3 * (i + j * this->_output_width);

      _frame_buffer[pixel_index] =
          static_cast<float>(clamp(ray_color.getX(), 0, 0.999));
      _frame_buffer[pixel_index + 1] =
          static_cast<float>(clamp(ray_color.getY(), 0, 0.999));
      _frame_buffer[pixel_index + 2] =
          static_cast<float>(clamp(ray_color.getZ(), 0, 0.999));
    }
  }
}

Color Renderer::getRayColor(const Ray &ray, const Scene &scene) {

  if (HitRecord rec; scene.hit(ray, 0.001, INFINITY, rec)) {
    return 0.5 * (rec.normal + Color(1, 1, 1));
  }

  return Color(0, 0, 0);
}
