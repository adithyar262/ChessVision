#pragma once

#include <GL/glew.h>

class Texture {
public:
    GLuint ID;

    Texture(const char* path);
    void bind();
};
