cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0135 NEW)

project(Raydiance)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Library

set(LIBRARY_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

set(LIBRARY_SOURCES
    ${LIBRARY_SOURCE_DIR}/3d/Camera.cpp
    ${LIBRARY_SOURCE_DIR}/3d/Object.cpp
    ${LIBRARY_SOURCE_DIR}/3d/Renderer.cpp
    ${LIBRARY_SOURCE_DIR}/3d/Scene.cpp
    ${LIBRARY_SOURCE_DIR}/3d/Sphere.cpp

    ${LIBRARY_SOURCE_DIR}/math/Ray.cpp
    ${LIBRARY_SOURCE_DIR}/math/Vec3.cpp
)

add_library(raydiance SHARED ${LIBRARY_SOURCES})
target_include_directories(
    raydiance
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/math
    ${CMAKE_SOURCE_DIR}/include/3d
)

install(
    TARGETS raydiance
    RUNTIME
    LIBRARY
)

# Executable

add_executable(demo src/main.cpp)
target_link_libraries(demo PUBLIC raydiance)

# Google Test setup

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Google Test

enable_testing()

add_executable(
    raydiance_test
    test/main.cpp
)
target_link_libraries(
    raydiance_test
    raydiance
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(raydiance_test)