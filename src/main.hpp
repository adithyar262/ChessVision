#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>
#include <unordered_map>

struct TextureCoord {
    float u, v;
};

std::unordered_map<char, TextureCoord> pieceToTexture = {
    {'R', {0.0f, 0.0f}},   // White Rook
    {'N', {0.125f, 0.0f}}, // White Knight
    {'B', {0.25f, 0.0f}},  // White Bishop
    {'Q', {0.375f, 0.0f}}, // White Queen
    {'K', {0.5f, 0.0f}},   // White King
    {'P', {0.0f, 0.125f}}, // White Pawn
    {'r', {0.0f, 0.875f}},   // Black Rook
    {'n', {0.125f, 0.875f}}, // Black Knight
    {'b', {0.25f, 0.875f}},  // Black Bishop
    {'q', {0.375f, 0.875f}}, // Black Queen
    {'k', {0.5f, 0.875f}},   // Black King
    {'p', {0.625f, 0.75f}},  // Black Pawn
    {' ', {0.75f, 0.5f}}   // Empty square
};

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