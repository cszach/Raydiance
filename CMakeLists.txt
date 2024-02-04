cmake_minimum_required(VERSION 3.5)

project(Raydiance LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library

set(LIBRARY_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

set(LIBRARY_SOURCES
    ${LIBRARY_SOURCE_DIR}/3d/Camera.cu
    ${LIBRARY_SOURCE_DIR}/3d/Material.cu
    ${LIBRARY_SOURCE_DIR}/3d/Object.cu
    ${LIBRARY_SOURCE_DIR}/3d/Renderer.cu
    ${LIBRARY_SOURCE_DIR}/3d/Scene.cu
    ${LIBRARY_SOURCE_DIR}/3d/Sphere.cu

    ${LIBRARY_SOURCE_DIR}/math/Ray.cu
    ${LIBRARY_SOURCE_DIR}/math/Vec3.cu

    ${LIBRARY_SOURCE_DIR}/cuda/cuda_helper.cu
)

add_library(raydiance STATIC ${LIBRARY_SOURCES})
target_include_directories(
    raydiance
    PUBLIC
    ${CMAKE_SOURCE_DIR}/include/3d
    ${CMAKE_SOURCE_DIR}/include/cuda
    ${CMAKE_SOURCE_DIR}/include/math
    /usr/local/cuda-12.2/targets/x86_64-linux/include
)

set_target_properties(raydiance PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "86")

install(
    TARGETS raydiance
    RUNTIME
    LIBRARY
)

# Executable

add_executable(demo src/main.cu)
target_link_libraries(demo PUBLIC raydiance)

set_target_properties(demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "86")
