#include <fstream>
#include <memory>

#include "Renderer.hpp"
#include "Scene.hpp"
#include "Sphere.hpp"

using std::make_shared;
using std::shared_ptr;

int main() {
  // Image

  const int IMAGE_WIDTH = 256;
  const int IMAGE_HEIGHT = 256;

  std::ofstream f_out("image.ppm");

  // Camera

  Camera camera(100, 1);

  // Scene

  Scene scene;
  shared_ptr<Sphere> test_sphere = make_shared<Sphere>(0.5);
  test_sphere->setPosition(Point3(0, 0, -1));

  scene.add(test_sphere);

  // Render

  Renderer renderer(IMAGE_WIDTH, IMAGE_HEIGHT);

  float *frame_buffer = renderer.getFrameBuffer();

  renderer.render(scene, camera);

  // Write to PPM

  f_out << "P3\n" << IMAGE_WIDTH << ' ' << IMAGE_HEIGHT << '\n' << 255 << '\n';

  for (int i = 0; i < IMAGE_WIDTH; i++) {
    for (int j = 0; j < IMAGE_HEIGHT; j++) {
      int pixel_index = 3 * (i + j * IMAGE_WIDTH);

      float r = frame_buffer[pixel_index + 0];
      float g = frame_buffer[pixel_index + 1];
      float b = frame_buffer[pixel_index + 2];

      int ir = int(255.99 * r);
      int ig = int(255.99 * g);
      int ib = int(255.99 * b);

      f_out << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }
}