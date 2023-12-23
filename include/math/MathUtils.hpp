#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <limits>
#include <random>

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degToRad(double deg) { return deg * pi / 180.0; }

inline double clamp(double value, double min, double max) {
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

inline double random_double() {
  static std::uniform_real_distribution distribution(0.0, 1.0);
  static std::mt19937 generator;

  return distribution(generator);
}

inline double random_double(double min, double max) {
  static std::uniform_real_distribution distribution(min, max);
  static std::mt19937 generator;

  return distribution(generator);
}

#endif // MATH_UTILS_H
