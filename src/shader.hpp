#pragma once

#include <GL/glew.h>
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
