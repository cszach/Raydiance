#include <gtest/gtest.h>

#include "Ray.hpp"

class RayTest : public ::testing::Test {
public:
  Ray default_ray;
  Ray r;

protected:
  void SetUp() override {
    default_ray = Ray();
    r = Ray(Point3(0, 1, 2), Vec3(3, 4, 5));
  }
};

TEST_F(RayTest, GettersAndSetters) {
  EXPECT_TRUE(default_ray.getOrigin().equals(Point3(0, 0, 0)));
  EXPECT_TRUE(default_ray.getDirection().equals(Vec3(0, 0, -1)));

  EXPECT_TRUE(r.getOrigin().equals(Point3(0, 1, 2)));
  EXPECT_TRUE(r.getDirection().equals(Vec3(3, 4, 5)));
}

TEST_F(RayTest, At) {
  EXPECT_TRUE(r.at(0).equals(Point3(0, 1, 2)));
  EXPECT_TRUE(r.at(1).equals(Point3(3, 5, 7)));
  EXPECT_TRUE(r.at(2).equals(Point3(6, 9, 12)));
}
