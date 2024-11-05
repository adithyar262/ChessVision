#include "renderer.hpp"
#include <iostream>

std::vector<std::vector<char>> parseFEN(const std::string& fen) {
    std::vector<std::vector<char>> board(8, std::vector<char>(8, ' '));
    int row = 0, col = 0;
    for (char c : fen) {
        if (c == '/') {
            //row++;
            col = 0;
        } else if (c >= '1' && c <= '8') {
            col += c - '0';
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            if (row >= 0 && row < 8 && col >= 0 && col < 8) {
                board[row][col] = c;
            }
            col++;
        }
        if (col >= 8) {
            row++;
            col = 0;
        }
        if (row >= 8) break;
    }
    return board;
}

Renderer::Renderer() {
    const float step = 1.0f / 8.0f;
    const int gridSize = 8;
    const int verticesPerSquare = 6;
    const int componentsPerVertex = 5;
    const int totalVertices = gridSize * gridSize * verticesPerSquare;
    float v1[totalVertices * componentsPerVertex];

    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            float x1 = -1.0f + (col * 2 * step);
            float x2 = x1 + 2 * step;
            float y1 = -1.0f + (row * 2 * step);
            float y2 = y1 + 2 * step;
            int baseIndex = (row * gridSize + col) * verticesPerSquare * componentsPerVertex;

            // First triangle
            v1[baseIndex     ] = x1; v1[baseIndex + 1 ] = y2; v1[baseIndex + 2 ] = 0.0f; v1[baseIndex + 3 ] = col * step; v1[baseIndex + 4 ] = (row + 1) * step;
            v1[baseIndex + 5 ] = x2; v1[baseIndex + 6 ] = y2; v1[baseIndex + 7 ] = 0.0f; v1[baseIndex + 8 ] = (col + 1) * step; v1[baseIndex + 9 ] = (row + 1) * step;
            v1[baseIndex + 10] = x2; v1[baseIndex + 11] = y1; v1[baseIndex + 12] = 0.0f; v1[baseIndex + 13] = (col + 1) * step; v1[baseIndex + 14] = row * step;

            // Second triangle
            v1[baseIndex + 15] = x1; v1[baseIndex + 16] = y2; v1[baseIndex + 17] = 0.0f; v1[baseIndex + 18] = col * step; v1[baseIndex + 19] = (row + 1) * step;
            v1[baseIndex + 20] = x2; v1[baseIndex + 21] = y1; v1[baseIndex + 22] = 0.0f; v1[baseIndex + 23] = (col + 1) * step; v1[baseIndex + 24] = row * step;
            v1[baseIndex + 25] = x1; v1[baseIndex + 26] = y1; v1[baseIndex + 27] = 0.0f; v1[baseIndex + 28] = col * step; v1[baseIndex + 29] = row * step;
        }
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v1), v1, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    board = std::vector<std::vector<char>>(8, std::vector<char>(8, ' '));
}

void Renderer::draw() {
    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 6*64);
}

void Renderer::updateBoard(const std::string& fen) {
    board = parseFEN(fen);
    std::cout << "Current board state:" << std::endl;
    for (int row = 7; row >= 0; --row) {
        for (int col = 0; col < 8; ++col) {
            std::cout << (board[row][col] == ' ' ? '.' : board[row][col]) << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

Renderer::~Renderer() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}
