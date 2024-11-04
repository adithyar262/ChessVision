#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

class Shader {
public:
    GLuint ID;

    Shader(const char* vertexPath, const char* fragmentPath);
    void use();

private:
    std::string readShaderFile(const char* filePath);
    GLuint compileShader(GLenum type, const char* source);
    void checkCompileErrors(GLuint shader, std::string type);
};

class Texture {
public:
    GLuint ID;

    Texture(const char* path);
    void bind();
};

class Renderer {
public:
    GLuint VAO, VBO;
    
    Renderer();
    void draw();
    ~Renderer();
};