cmake_minimum_required(VERSION 3.18)
project(chess_vision LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# TensorRT configuration
set(TensorRT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT" CACHE PATH "Path to TensorRT")
include_directories("${TensorRT_DIR}/include")
link_directories("${TensorRT_DIR}/lib")

if(OpenCV_FOUND)
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found. Please check OpenCV_DIR.")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

if(UNIX)
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_LIBRARIES})
else(UNIX)
    set(EXTERNAL "${CMAKE_SOURCE_DIR}/external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})

    set(LIBRARIES ${LIBRARIES} ws2_32)
endif(UNIX)

set(GLM_ROOT_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})

set(STB_ROOT_DIR "${CMAKE_SOURCE_DIR}/external/include/stb")
include_directories(${STB_ROOT_DIR})

# Add include directories for our source files
include_directories(
    ${CMAKE_SOURCE_DIR}/src
)

set(ALL_HEADERS
    src/shader.hpp
    src/texture.hpp
    src/renderer.hpp
)

set(ALL_SOURCES
    src/main.cpp
    src/shader.cpp
    src/texture.cpp
    src/renderer.cpp
)

source_group("Header Files" FILES ${ALL_HEADERS})
source_group("Source Files" FILES ${ALL_SOURCES})

# Create executable
add_executable(${CMAKE_PROJECT_NAME} ${ALL_HEADERS} ${ALL_SOURCES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${LIBRARIES} ${OpenCV_LIBS})

# CUDA architecture settings
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
endif()

# CUDA debug options
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G>")

# Set as startup project in Visual Studio
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

# Copy shaders to build directory
add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/bin/shaders
)

# Copy textures to build directory
add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/textures
        ${CMAKE_BINARY_DIR}/bin/textures
)
