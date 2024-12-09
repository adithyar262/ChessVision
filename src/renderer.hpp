#pragma once

#include <GL/glew.h>
#include <vector>
#include <string>

class Renderer {
public:
    GLuint VAO, VBO;
    std::vector<std::vector<char>> board;
    
    Renderer();
    void draw();
    void updateBoard(const std::string& fen);

    void UpdateVertices(float score);
    ~Renderer();
};

std::vector<std::vector<char>> parseFEN(const std::string& fen);
