#include <gtest/gtest.h>

#include <cmath>

#include "Camera.hpp"
#include "MathUtils.hpp"

class CameraTest : public ::testing::Test {
protected:
  const double vertical_fov = 45.0;
  const double theta = degToRad(vertical_fov);
  const double h = tan(theta / 2.0);
  const double aspect_ratio = 16.0 / 9.0;
  const double viewport_height = 2.0 * h;
  const double viewport_width = viewport_height * aspect_ratio;
  const double focal_length = 1.0;
  const Point3 position = Point3(0, 1, 2);

  Camera camera;
  Camera camera_50;
  Camera default_camera;

  void SetUp() override {
    camera = Camera(vertical_fov, aspect_ratio);
    camera_50 = Camera(50, aspect_ratio);
    default_camera = Camera();

    camera.setPosition(position);
    camera_50.setFocalLength(2);
  }
};

TEST_F(CameraTest, GettersAndSetters) {
  EXPECT_EQ(camera.getVerticalFOV(), vertical_fov);
  EXPECT_EQ(camera.getAspectRatio(), aspect_ratio);
  EXPECT_EQ(camera.getViewportHeight(), 2.0 * h);
  EXPECT_EQ(camera.getViewportWidth(), camera.getViewportHeight() * 16.0 / 9.0);
  EXPECT_EQ(camera.getFocalLength(), 1.0);
  EXPECT_TRUE(camera.getPosition().equals(position));

  camera.setVerticalFov(50);

  EXPECT_EQ(camera.getVerticalFOV(), camera_50.getVerticalFOV());
  EXPECT_EQ(camera.getViewportHeight(), camera_50.getViewportHeight());
  EXPECT_EQ(camera.getViewportWidth(), camera_50.getViewportWidth());

  camera.setAspectRatio(1.0);

  EXPECT_EQ(camera.getAspectRatio(), default_camera.getAspectRatio());
  EXPECT_EQ(camera.getViewportHeight(), default_camera.getViewportHeight());
  EXPECT_EQ(camera.getViewportWidth(), default_camera.getViewportWidth());

  camera.setFocalLength(camera_50.getFocalLength());
  EXPECT_EQ(camera.getFocalLength(), camera_50.getFocalLength());

  camera.setPosition(Point3(0, 0, 0));
  EXPECT_TRUE(camera.getPosition().equals(default_camera.getPosition()));
}