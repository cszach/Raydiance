#ifndef PPM_OUTPUT_H
#define PPM_OUTPUT_H

#include <iostream>

#include "ImageOutput.hpp"

class PPMOutput : public ImageOutput {
private:
  std::ostream &output_stream;

public:
  PPMOutput() : output_stream(std::cout) {}
  PPMOutput(std::ostream &output_stream) : output_stream(output_stream) {}

  void writeHeader(int image_width, int image_height, int max_color) {
    this->output_stream << "P3\n"
                        << image_width << ' ' << image_height << '\n'
                        << max_color << '\n';
  }

  inline void writeColor(int r, int g, int b) const override {
    this->output_stream << r << ' ' << g << ' ' << b << '\n';
  }
};

#endif // PPM_OUTPUT_H