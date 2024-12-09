cmake_minimum_required(VERSION 3.10)
project(ChessVision CUDA CXX)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/external/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Add CUDA source files
cuda_add_library(cuda_kernels
    src/cuda_kernel.cu
)

# Add executable
add_executable(chessvision
    src/main.cpp
    src/shader.cpp
    src/texture.cpp
    src/renderer.cpp
)

# Link libraries
target_link_libraries(chessvision
    cuda_kernels
    ${OPENGL_LIBRARIES}
    GLEW
    glfw
    ${CUDA_LIBRARIES}
)
