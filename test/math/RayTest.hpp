#include <gtest/gtest.h>

#include "Ray.hpp"

class RayTest : public ::testing::Test {
protected:
  Ray default_ray;
  Ray r;

  void SetUp() override {
    default_ray = Ray();
    r = Ray(Point3(0, 1, 2), Vec3(3, 4, 5));
  }
};

TEST_F(RayTest, GettersAndSetters) {
  EXPECT_EQ(default_ray.getOrigin(), Point3(0, 0, 0));
  EXPECT_EQ(default_ray.getDirection(), Vec3(0, 0, -1));

  EXPECT_EQ(r.getOrigin(), Point3(0, 1, 2));
  EXPECT_EQ(r.getDirection(), Vec3(3, 4, 5));
}

TEST_F(RayTest, At) {
  EXPECT_EQ(r.at(0), Point3(0, 1, 2));
  EXPECT_EQ(r.at(1), Point3(3, 5, 7));
  EXPECT_EQ(r.at(2), Point3(6, 9, 12));
}