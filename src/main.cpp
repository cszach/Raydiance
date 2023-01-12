#include <fstream>
#include <ostream>

#include "PPMOutput.hpp"
#include "Renderer.hpp"

int main() {
  std::ofstream f_out("image.ppm");
  std::ostream &f_ostream(f_out);

  f_ostream.rdbuf(f_out.rdbuf());

  PPMOutput output(f_ostream);

  // Image

  const int IMAGE_WIDTH = 256;
  const int IMAGE_HEIGHT = 256;

  output.writeHeader(IMAGE_WIDTH, IMAGE_HEIGHT, 255);

  for (int i = 0; i < IMAGE_WIDTH; i++) {
    for (int j = 0; j < IMAGE_HEIGHT; j++) {
      int r = static_cast<int>(255.999 * static_cast<double>(i) /
                               (IMAGE_WIDTH - 1));
      int g = static_cast<int>(255.999 * static_cast<double>(j) /
                               (IMAGE_WIDTH - 1));
      int b = static_cast<int>(255.99 * 0.25);

      output.writeColor(r, g, b);
    }
  }
}