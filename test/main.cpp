#include "gtest/gtest.h"

#include "math/RayTest.hpp"
#include "math/Vec3Test.hpp"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
