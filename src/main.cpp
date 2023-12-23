#include <fstream>
#include <memory>

#include "Renderer.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"

using std::make_shared;
using std::shared_ptr;

int main() {
  // Image

  const double ASPECT_RATIO = 16.0 / 9.0;
  const int IMAGE_WIDTH = 400;
  const auto IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / ASPECT_RATIO);

  std::ofstream f_out("image.ppm");

  // Camera

  Camera camera(100, ASPECT_RATIO);

  // Scene

  Scene scene;
  auto test_sphere = make_shared<Sphere>(0.5);
  test_sphere->setPosition(Point3(0, 0, -1));

  scene.add(test_sphere);

  // Render

  Renderer renderer(camera, IMAGE_WIDTH, IMAGE_HEIGHT);

  renderer.render(scene);

  // Write to PPM
  const std::vector<float> frame_buffer = renderer.getFrameBuffer();

  f_out << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << '\n' << 255 << '\n';

  for (int j = 0; j < IMAGE_HEIGHT; ++j) {
    for (int i = 0; i < IMAGE_WIDTH; ++i) {
      int pixel_index = 3 * (i + j * IMAGE_WIDTH);

      float r = frame_buffer[pixel_index + 0];
      float g = frame_buffer[pixel_index + 1];
      float b = frame_buffer[pixel_index + 2];

      auto ir = int(255.99 * r);
      auto ig = int(255.99 * g);
      auto ib = int(255.99 * b);

      f_out << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }
}
