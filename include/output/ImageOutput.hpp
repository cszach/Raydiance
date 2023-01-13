#ifndef IMAGE_OUTPUT_H
#define IMAGE_OUTPUT_H

class ImageOutput {
public:
  virtual int getWidth() const = 0;
  virtual int getHeight() const = 0;
  virtual void writeColor(int r, int g, int b) const = 0;
};

#endif // IMAGE_OUTPUT_H